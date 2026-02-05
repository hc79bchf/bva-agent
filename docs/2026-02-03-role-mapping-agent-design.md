# Role Mapping Agent Design

**Date:** 2026-02-03
**Status:** Draft
**Author:** Human + Claude

## Overview

Replace the current fuzzy string matching for role-to-O*NET mapping with an LLM-enabled agent that provides semantic understanding. Store the full O*NET database locally for faster, more reliable candidate retrieval.

## Problem

The current role mapping system uses `SequenceMatcher.ratio()` (fuzzy string matching) to map uploaded job titles to O*NET occupations. This produces poor results because:

- No semantic understanding ("Product Manager" may match "Production Manager")
- Relies on character similarity, not job function
- Users frequently need to manually remap roles

## Solution

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Role Mapping Flow                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Upload CSV ──► Extract Roles ──► For each role:               │
│                                          │                      │
│                                          ▼                      │
│                        ┌─────────────────────────────┐          │
│                        │   Local O*NET Database      │          │
│                        │   (923 occupations, full)   │          │
│                        └──────────────┬──────────────┘          │
│                                       │                         │
│                          Keyword search (top 20)                │
│                                       │                         │
│                                       ▼                         │
│                        ┌─────────────────────────────┐          │
│                        │   RoleMappingAgent (LLM)    │          │
│                        │   - Batches of 10-15 roles  │          │
│                        │   - Returns code + tier     │          │
│                        └──────────────┬──────────────┘          │
│                                       │                         │
│                                       ▼                         │
│                        Store mappings with confidence           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| LLM invocation | Replace fuzzy matching entirely | Semantic understanding for all roles |
| Candidate source | LLM selects from O*NET candidates | Constrains to valid codes, no hallucination |
| Confidence output | Tiered (HIGH/MEDIUM/LOW) | LLMs better at categorical than numerical |
| O*NET data | Full local database | Faster, more reliable, richer context |
| Candidate retrieval | Keyword search on local DB | Simple, alternate titles handle synonyms |
| Batching | Chunks of 10-15 roles | Balance of cost, speed, reliability |

## Database Schema

### O*NET Tables

```sql
-- Core occupation table
onet_occupations (
    code VARCHAR(10) PRIMARY KEY,     -- "15-1252.00"
    title VARCHAR(255) NOT NULL,       -- "Software Developers"
    description TEXT,                  -- Full description
    updated_at TIMESTAMP
)

-- Alternate titles for better keyword matching
onet_alternate_titles (
    id UUID PRIMARY KEY,
    onet_code VARCHAR(10) REFERENCES onet_occupations(code),
    title VARCHAR(255) NOT NULL,       -- "Programmer", "Coder", etc.
)

-- Tasks (what the job does)
onet_tasks (
    id UUID PRIMARY KEY,
    onet_code VARCHAR(10) REFERENCES onet_occupations(code),
    description TEXT NOT NULL,
    importance FLOAT                   -- 1-5 scale from O*NET
)

-- Sync tracking
onet_sync_log (
    id UUID PRIMARY KEY,
    version VARCHAR(20),               -- "30.1"
    synced_at TIMESTAMP,
    occupation_count INT,
    status VARCHAR(20)                 -- "success" / "failed"
)
```

### Indexes

```sql
-- Full-text search on titles
CREATE INDEX idx_onet_title_search ON onet_occupations USING gin(to_tsvector('english', title || ' ' || description));
CREATE INDEX idx_alt_title_search ON onet_alternate_titles USING gin(to_tsvector('english', title));
```

## Component Design

### RoleMappingAgent

```python
class RoleMappingAgent:
    """LLM-powered agent for semantic role-to-O*NET mapping."""

    def __init__(self, llm_service: LLMService, onet_repository: OnetRepository):
        self.llm = llm_service
        self.onet_repo = onet_repository
        self.batch_size = 12  # roles per LLM call
        self.candidates_per_role = 20

    async def map_roles(self, roles: list[str]) -> list[RoleMappingResult]:
        """Map a list of role titles to O*NET occupations."""
        # 1. Retrieve candidates for all roles
        role_candidates = await self._get_candidates(roles)

        # 2. Chunk into batches
        batches = self._chunk(roles, role_candidates, self.batch_size)

        # 3. Process batches
        results = []
        for batch in batches:
            batch_results = await self._process_batch(batch)
            results.extend(batch_results)

        return results

    async def _get_candidates(self, roles: list[str]) -> dict[str, list[OnetOccupation]]:
        """Retrieve top candidates for each role via keyword search."""
        candidates = {}
        for role in roles:
            candidates[role] = await self.onet_repo.search(
                query=role,
                limit=self.candidates_per_role
            )
        return candidates

    async def _process_batch(self, batch: list[RoleBatch]) -> list[RoleMappingResult]:
        """Process a batch of roles through the LLM."""
        prompt = self._build_prompt(batch)
        response = await self.llm.generate_response(
            system_prompt=ROLE_MAPPING_SYSTEM_PROMPT,
            user_message=prompt
        )
        return self._parse_response(response, batch)
```

### LLM Prompt

```python
ROLE_MAPPING_SYSTEM_PROMPT = """You are an expert at mapping job titles to O*NET occupations.

For each role provided, select the best matching O*NET occupation from the candidates listed.

Return your confidence level:
- HIGH: Clear, unambiguous match
- MEDIUM: Reasonable match but some ambiguity
- LOW: Best guess, significant uncertainty

Respond with a JSON array containing objects with these fields:
- role: The original role title
- onet_code: The selected O*NET code
- onet_title: The selected O*NET title
- confidence: HIGH, MEDIUM, or LOW
- reasoning: Brief explanation (1 sentence)
"""
```

### Output Schema

```python
class RoleMappingResult:
    source_role: str
    onet_code: str | None
    onet_title: str | None
    confidence: Literal["HIGH", "MEDIUM", "LOW"]
    confidence_score: float  # HIGH=0.95, MEDIUM=0.75, LOW=0.50
    reasoning: str
```

### Confidence Score Mapping

| Tier | Score | UI Treatment |
|------|-------|--------------|
| HIGH | 0.95 | Auto-confirmable, green badge |
| MEDIUM | 0.75 | Needs review, yellow badge |
| LOW | 0.50 | Likely wrong, red badge |

### OnetSyncService

```python
class OnetSyncService:
    """Downloads and imports O*NET database releases."""

    ONET_DOWNLOAD_URL = "https://www.onetcenter.org/dl_files/database/"

    async def sync(self, version: str = "latest") -> SyncResult:
        """Download and import O*NET data."""
        # 1. Download zip file
        zip_path = await self._download(version)

        # 2. Extract relevant CSVs
        files = self._extract([
            "Occupation Data.txt",
            "Alternate Titles.txt",
            "Task Statements.txt"
        ])

        # 3. Parse and upsert to database
        counts = await self._import_data(files)

        # 4. Log sync
        await self._log_sync(version, counts)

        return SyncResult(version=version, **counts)
```

**Sync triggers:**
- Manual: `POST /admin/onet/sync`
- Startup: Check if DB empty or stale (>90 days)
- Future: Scheduled monthly check

## Error Handling

| Failure | Behavior |
|---------|----------|
| LLM timeout/error on batch | Retry 2x, then mark roles as LOW confidence with `reasoning: "Mapping failed, please remap manually"` |
| No candidates found | Return LOW confidence, `onet_code: null`, prompt user to remap |
| O*NET DB empty | Block mapping, return error "O*NET data not synced" |
| Invalid LLM response | Retry once, then fall back to first candidate with LOW confidence |

**Principle:** Never block the user. Provide low-confidence fallback so user can manually fix.

## Files to Create/Modify

### New Files

| File | Purpose |
|------|---------|
| `app/services/role_mapping_agent.py` | LLM-based mapping agent |
| `app/services/onet_sync_service.py` | O*NET data sync |
| `app/repositories/onet_repository.py` | O*NET database queries |
| `app/models/onet.py` | SQLAlchemy models for O*NET tables |
| `migrations/xxx_onet_tables.py` | Database migration |

### Modified Files

| File | Changes |
|------|---------|
| `app/services/role_mapping_service.py` | Use agent instead of fuzzy matcher |
| `app/routers/role_mappings.py` | Add admin sync endpoint |
| `app/services/__init__.py` | Export new services |

## Testing Strategy

1. **Unit tests:** Mock LLM responses, verify batch chunking and parsing
2. **Integration tests:** Test full flow with real O*NET data (subset)
3. **Manual QA:** Upload sample CSVs, verify mapping quality improvement

## Future Enhancements

- **Embedding search:** Add vector similarity for better candidate retrieval
- **Feedback loop:** Learn from user corrections to improve prompts
- **Caching:** Cache LLM results for identical role titles
