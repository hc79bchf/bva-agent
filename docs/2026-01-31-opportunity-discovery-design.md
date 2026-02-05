# Opportunity Discovery Design

> **For Claude:** This design document captures the architecture for Phase 0 - Opportunity Discovery. Use this as the source of truth when implementing the feature.

**Goal:** Enable enterprises to identify automation opportunities by mapping their workforce roles to O*NET occupations and analyzing work activities for AI/automation potential.

**Architecture:** Orchestrator + 5 specialized subagents managing a 5-step wizard workflow with integrated chat. O*NET data synced weekly via API, with GWA-level AI exposure scores inherited down to DWA level and refinable by users.

**Tech Stack:** FastAPI backend, React 18+ frontend with shadcn/ui, PostgreSQL with O*NET data tables, S3 for file storage.

---

## Table of Contents

1. [Data Model](#1-data-model)
2. [O*NET API Integration](#2-onet-api-integration)
3. [Scoring Engine](#3-scoring-engine)
4. [Workflow](#4-workflow)
5. [Agent Architecture](#5-agent-architecture)
6. [Per-Agent Memory](#6-per-agent-memory)
7. [Error Handling](#7-error-handling)
8. [Frontend/UI Design](#8-frontendui-design)

---

## 1. Data Model

### O*NET Reference Tables (Read-Only, Synced Weekly)

```
onet_occupations (923 records)
├── code: VARCHAR(10) PK        # e.g., "15-1252.00"
├── title: VARCHAR(255)          # e.g., "Software Developers"
├── description: TEXT
└── updated_at: TIMESTAMP

onet_gwa (41 records) - Generalized Work Activities
├── id: VARCHAR(20) PK           # e.g., "4.A.1.a.1"
├── name: VARCHAR(255)           # e.g., "Getting Information"
├── description: TEXT
├── ai_exposure_score: FLOAT     # 0.0-1.0, from Pew Research mapping
└── updated_at: TIMESTAMP

onet_iwa (~300 records) - Intermediate Work Activities
├── id: VARCHAR(20) PK
├── gwa_id: VARCHAR(20) FK
├── name: VARCHAR(255)
├── description: TEXT
└── updated_at: TIMESTAMP

onet_dwa (2000+ records) - Detailed Work Activities
├── id: VARCHAR(20) PK
├── iwa_id: VARCHAR(20) FK
├── name: VARCHAR(255)
├── description: TEXT
├── ai_exposure_override: FLOAT  # NULL = inherit from GWA
└── updated_at: TIMESTAMP

onet_tasks (~19,000 records)
├── id: SERIAL PK
├── occupation_code: VARCHAR(10) FK
├── description: TEXT
├── importance: FLOAT
└── updated_at: TIMESTAMP

onet_task_to_dwa (junction table)
├── task_id: INT FK
└── dwa_id: VARCHAR(20) FK

onet_skills
├── id: VARCHAR(20) PK
├── name: VARCHAR(255)
├── description: TEXT
└── updated_at: TIMESTAMP

onet_technology_skills
├── id: SERIAL PK
├── occupation_code: VARCHAR(10) FK
├── technology_name: VARCHAR(255)
├── hot_technology: BOOLEAN
└── updated_at: TIMESTAMP
```

### Application Tables

```
discovery_sessions
├── id: UUID PK
├── user_id: UUID FK
├── organization_id: UUID FK
├── status: ENUM(draft, in_progress, completed, archived)
├── current_step: INT            # 1-5
├── created_at: TIMESTAMP
└── updated_at: TIMESTAMP

discovery_uploads
├── id: UUID PK
├── session_id: UUID FK
├── file_name: VARCHAR(255)
├── file_url: VARCHAR(512)       # S3 path
├── row_count: INT
├── column_mappings: JSONB       # {role: "Column B", department: "Column C", ...}
├── detected_schema: JSONB
└── created_at: TIMESTAMP

discovery_role_mappings
├── id: UUID PK
├── session_id: UUID FK
├── source_role: VARCHAR(255)    # From uploaded file
├── onet_code: VARCHAR(10) FK
├── confidence_score: FLOAT
├── user_confirmed: BOOLEAN
├── row_count: INT               # How many rows have this role
└── created_at: TIMESTAMP

discovery_activity_selections
├── id: UUID PK
├── session_id: UUID FK
├── role_mapping_id: UUID FK
├── dwa_id: VARCHAR(20) FK
├── selected: BOOLEAN
├── user_modified: BOOLEAN       # Did user change from default?
└── created_at: TIMESTAMP

discovery_analysis_results
├── id: UUID PK
├── session_id: UUID FK
├── role_mapping_id: UUID FK
├── dimension: ENUM(role, task, lob, geography, department)
├── dimension_value: VARCHAR(255)
├── ai_exposure_score: FLOAT
├── impact_score: FLOAT
├── complexity_score: FLOAT
├── priority_score: FLOAT
├── breakdown: JSONB             # Detailed scoring breakdown
└── created_at: TIMESTAMP

agentification_candidates
├── id: UUID PK
├── session_id: UUID FK
├── role_mapping_id: UUID FK
├── name: VARCHAR(255)
├── description: TEXT
├── priority_tier: ENUM(now, next_quarter, future)
├── estimated_impact: FLOAT
├── selected_for_build: BOOLEAN
├── intake_request_id: UUID FK   # NULL until sent to Build
└── created_at: TIMESTAMP
```

---

## 2. O*NET API Integration

### Sync Strategy

- **Frequency:** Weekly job (Sunday 2am UTC)
- **Approach:** Full refresh with soft deletes
- **API Base:** `https://services.onetcenter.org/ws/`
- **Authentication:** HTTP Basic Auth (API key as username)

### Endpoints Used

| Endpoint | Data | Records |
|----------|------|---------|
| `/mnm/search?keyword=` | Occupation search | 923 |
| `/online/occupations/{code}` | Occupation details | Per occupation |
| `/online/occupations/{code}/summary/tasks` | Tasks by occupation | ~19,000 total |
| `/online/occupations/{code}/summary/work_activities` | DWAs by occupation | Links |
| `/online/occupations/{code}/summary/skills` | Skills by occupation | Links |
| `/online/occupations/{code}/summary/technology_skills` | Tech skills | Links |

### Sync Job Implementation

```python
class OnetSyncJob:
    """Weekly sync of O*NET data to local database."""

    async def run(self):
        # 1. Fetch all occupations
        occupations = await self.fetch_all_occupations()

        # 2. For each occupation, fetch details
        for occ in occupations:
            tasks = await self.fetch_tasks(occ.code)
            activities = await self.fetch_work_activities(occ.code)
            skills = await self.fetch_skills(occ.code)
            tech_skills = await self.fetch_technology_skills(occ.code)

            await self.upsert_occupation_data(occ, tasks, activities, skills, tech_skills)

        # 3. Build GWA → IWA → DWA hierarchy from activity data
        await self.build_activity_hierarchy()

        # 4. Apply Pew Research AI exposure scores to GWA level
        await self.apply_baseline_exposure_scores()
```

### Rate Limiting

- O*NET API: 10 requests/second limit
- Implement exponential backoff on 429 responses
- Full sync takes ~2-3 hours

---

## 3. Scoring Engine

### AI Exposure Score Calculation

```python
def calculate_exposure_score(role_mapping: RoleMapping) -> float:
    """
    Calculate AI exposure score for a role based on selected DWAs.

    Score flows: GWA (baseline) → IWA → DWA (with optional override)
    """
    selected_dwas = get_selected_dwas(role_mapping)

    scores = []
    for dwa in selected_dwas:
        if dwa.ai_exposure_override is not None:
            score = dwa.ai_exposure_override
        else:
            # Inherit from parent GWA
            gwa = dwa.iwa.gwa
            score = gwa.ai_exposure_score
        scores.append(score)

    return sum(scores) / len(scores) if scores else 0.0
```

### Pew Research GWA Mapping

The 41 Generalized Work Activities are pre-scored based on Pew Research AI exposure analysis:

| GWA Category | Example Activities | Base Score |
|--------------|-------------------|------------|
| Information Input | Getting Information, Monitoring Processes | 0.7-0.9 |
| Mental Processes | Analyzing Data, Making Decisions | 0.6-0.8 |
| Work Output | Documenting/Recording, Handling Objects | 0.3-0.7 |
| Interacting | Communicating, Coordinating | 0.4-0.6 |

### Multi-Dimensional Scoring

```python
@dataclass
class AnalysisScores:
    ai_exposure: float      # 0-1, from DWA aggregation
    impact: float           # 0-1, based on role count * exposure
    complexity: float       # 0-1, inverse of exposure (harder to automate)
    priority: float         # 0-1, weighted combination

def calculate_priority(exposure: float, impact: float, complexity: float) -> float:
    """Priority = high exposure + high impact + low complexity."""
    return (exposure * 0.4) + (impact * 0.4) + ((1 - complexity) * 0.2)
```

---

## 4. Workflow

### 5-Step Wizard with Chat Checkpoints

```
Step 1: Upload
├── User uploads HR data file (.xlsx, .csv)
├── Agent parses file, detects columns
├── Agent asks clarifying questions about column meanings
└── Checkpoint: Column mappings confirmed

Step 2: Map Roles
├── Agent extracts unique roles from mapped column
├── Agent matches each role to O*NET occupations
├── User reviews/adjusts mappings with confidence scores
└── Checkpoint: All roles mapped to O*NET codes

Step 3: Activities
├── Agent loads default DWAs for each mapped occupation
├── User reviews/adjusts activity selections per role
├── Changes tracked for learning
└── Checkpoint: Activity selections confirmed

Step 4: Analysis
├── Agent calculates scores across all dimensions
├── User explores data via 5 view tabs:
│   ├── By Role
│   ├── By Task
│   ├── By Line of Business
│   ├── By Geography
│   └── By Department
└── Checkpoint: User understands opportunities

Step 5: Roadmap
├── Agent proposes prioritized timeline
├── User adjusts via drag-and-drop Kanban
├── User selects candidates to send to Build
└── Checkpoint: Handoff to intake complete
```

### Step Transitions

- Users can move backward to any previous step
- Moving backward doesn't lose data (edits preserved)
- Moving forward requires checkpoint completion
- Agent prompts if user tries to skip required input

---

## 5. Agent Architecture

### Orchestrator + Subagent Pattern

```
Discovery Orchestrator
├── Manages single conversation thread with user
├── Routes to appropriate subagent based on current step
├── Maintains session state and context
├── Handles cross-cutting concerns (errors, help requests)
│
├── Upload Subagent
│   ├── File parsing (pandas/openpyxl)
│   ├── Column type detection
│   ├── Schema validation
│   └── Clarifying questions about data
│
├── Mapping Subagent
│   ├── Role text → O*NET search
│   ├── Confidence scoring
│   ├── Bulk operations
│   └── Alternative suggestions
│
├── Activity Subagent
│   ├── DWA loading by occupation
│   ├── Selection management
│   ├── Activity explanation
│   └── Default recommendations
│
├── Analysis Subagent
│   ├── Score calculations
│   ├── Dimension aggregations
│   ├── Insight generation
│   └── Chart data preparation
│
└── Roadmap Subagent
    ├── Prioritization logic
    ├── Timeline generation
    ├── Handoff bundle creation
    └── Build workflow integration
```

### Subagent Invocation

```python
class DiscoveryOrchestrator:
    """Routes user messages to appropriate subagent."""

    subagents = {
        1: UploadSubagent,
        2: MappingSubagent,
        3: ActivitySubagent,
        4: AnalysisSubagent,
        5: RoadmapSubagent,
    }

    async def handle_message(self, session: DiscoverySession, message: str):
        subagent_class = self.subagents[session.current_step]
        subagent = subagent_class(session, self.memory)

        response = await subagent.process(message)

        # Check if step is complete
        if response.checkpoint_complete:
            session.current_step += 1
            await self.announce_step_transition(session)

        return response
```

### Protocol Capabilities (Future-Ready)

Each agent is architected to support:
- **MCP (Model Context Protocol):** Disabled by default, `mcp_enabled: bool` flag
- **A2A (Agent-to-Agent):** Disabled by default, `a2a_enabled: bool` flag
- **A2UI (Agent-to-UI):** Disabled by default, `a2ui_enabled: bool` flag

```python
class BaseSubagent:
    mcp_enabled: bool = False
    a2a_enabled: bool = False
    a2ui_enabled: bool = False

    # Flags can be enabled per-agent when protocols are implemented
```

---

## 6. Per-Agent Memory

### Memory Model

Each subagent maintains its own memory for continuous learning:

```python
class AgentMemory:
    """Per-agent memory for learning within domain."""

    # Working memory - current session context
    working: dict  # Cleared after session

    # Episodic memory - specific interactions
    episodic: List[Episode]  # "User corrected X to Y"

    # Semantic memory - learned patterns
    semantic: List[Fact]  # "Role 'Software Engineer' usually maps to 15-1252.00"
```

### Memory Storage

```
agent_memory_working
├── agent_type: VARCHAR(50)      # e.g., "mapping_subagent"
├── session_id: UUID FK
├── context: JSONB
└── expires_at: TIMESTAMP

agent_memory_episodic
├── id: UUID PK
├── agent_type: VARCHAR(50)
├── organization_id: UUID FK     # Scoped to org for privacy
├── episode_type: VARCHAR(50)    # e.g., "role_mapping_correction"
├── content: JSONB               # {original: X, corrected: Y, context: ...}
├── created_at: TIMESTAMP
└── relevance_score: FLOAT       # Decays over time

agent_memory_semantic
├── id: UUID PK
├── agent_type: VARCHAR(50)
├── organization_id: UUID FK
├── fact_type: VARCHAR(50)       # e.g., "role_mapping_pattern"
├── content: JSONB               # {role_pattern: ".*Engineer.*", onet: "15-1252.00"}
├── confidence: FLOAT
├── occurrence_count: INT
└── last_updated: TIMESTAMP
```

### Learning Examples

| Agent | Learns From | Applies To |
|-------|-------------|------------|
| Upload | Column name patterns | Future file parsing |
| Mapping | User corrections to O*NET matches | Better initial suggestions |
| Activity | User DWA selections/deselections | Role-specific defaults |
| Analysis | Which views users explore most | Highlight relevant insights |
| Roadmap | Priority tier adjustments | Better initial prioritization |

---

## 7. Error Handling

### Graceful Degradation

| Failure | Fallback |
|---------|----------|
| O*NET API down | Use cached local data, warn "data may be stale" |
| File parse fails | Show error, suggest format fixes, allow retry |
| No O*NET match | Allow manual entry, flag for review |
| LLM timeout | Retry with exponential backoff, offer "try again" |
| Session lost | Auto-save every interaction, resume from last state |

### User-Facing Error Messages

```python
ERROR_MESSAGES = {
    "file_parse_error": "I couldn't read that file. Please check it's a valid .xlsx or .csv file.",
    "no_onet_match": "I couldn't find an O*NET match for '{role}'. Would you like to search manually?",
    "api_timeout": "That took longer than expected. Let me try again.",
    "session_expired": "Your session expired, but I saved your progress. Resuming from step {step}.",
}
```

### Recovery Actions

- All user inputs saved immediately (not just at checkpoints)
- Session state recoverable for 30 days
- Export option available at any step (partial results OK)

---

## 8. Frontend/UI Design

### Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│  Sidebar (240px)  │  Main Content (fluid, max 1280px)               │
│  ─────────────────│─────────────────────────────────────────────────│
│  [Logo]           │  ┌─────────────────────────────────────────────┐│
│                   │  │ Step Indicator (horizontal)                 ││
│  Discovery        │  │ ○ Upload → ○ Map → ○ Activities → ...      ││
│  • New Session    │  ├─────────────────────────────────────────────┤│
│  • Past Sessions  │  │                                             ││
│                   │  │  Step Content Area                          ││
│  ───────────────  │  │  (cards, tables, forms per step)           ││
│  Build Workflow   │  │                                             ││
│  Agent Registry   │  ├─────────────────────────────────────────────┤│
│  ...              │  │ Chat Panel (collapsible, 320px when open)   ││
│                   │  │ Docked to bottom-right                      ││
│                   │  └─────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

### Design System

- **Dark mode default:** `#0a0a0a` background, `#111111` for cards
- **Components:** shadcn/ui (Button, Card, Table, Dialog, Tabs)
- **Icons:** Lucide React
- **Typography:** Inter font, 12-30px type scale
- **Spacing:** 4px base unit (4, 8, 12, 16, 24, 32, 48)

### Step-by-Step Components

**Step 1: Upload**
- Drag-and-drop zone (dashed border card)
- Format badges: `.xlsx`, `.csv`
- Preview table (first 5 rows)

**Step 2: Map Roles**
- Two-column layout: Your Roles → O*NET Matches
- Role cards with confidence percentages
- Bulk actions: "Accept all >85%"

**Step 3: Activities**
- Expandable accordion by role
- DWA checklist grouped by IWA category
- Selection count badge

**Step 4: Analysis**
- Tab bar: By Role | By Task | By LoB | By Geography | By Department
- Sortable data table with sparklines
- Click-to-expand detail panel

**Step 5: Roadmap**
- Kanban timeline: Now | Next Quarter | Future
- Draggable candidate cards
- "Send to Build" action button

### Chat Panel

- Floating panel, bottom-right corner
- Collapsed: Icon button with unread indicator
- Expanded: 320px × 400px, resizable
- Background: `#111111`, 1px `#262626` border
- Quick action chips for common responses
- Keyboard: `Cmd+/` to toggle

### Brainstorming-Style Interactions

All agent interactions follow these principles:

| Principle | Implementation |
|-----------|----------------|
| One question at a time | Agent never asks multiple questions in one message |
| Multiple choice preferred | Quick action chips appear for choice questions |
| Progressive disclosure | Start simple, reveal detail on demand |
| Incremental validation | Agent confirms: "So you want X, correct?" |

**Example Flow:**

```
Agent: "I see your file has 12 columns. Which one contains job titles?"
       [Column A: Title] [Column B: Role] [Column D: Position] [Other...]

User: clicks [Column B: Role]

Agent: "Got it—using 'Role' for job titles. I found 47 unique roles.
        Should I start matching these to O*NET occupations?"
       [Yes, start matching] [Let me review the roles first]
```

### Chat + UI Coordination

- When agent asks a question, relevant UI section highlights
- User can answer via chat OR by interacting with UI
- Either method advances the conversation

### Output Persistence

| Agent | Artifacts | Storage |
|-------|-----------|---------|
| Orchestrator | Conversation transcript | `discovery_sessions` |
| Upload | Parsed file, mappings | `discovery_uploads` + S3 |
| Mapping | Role → O*NET links | `discovery_role_mappings` |
| Activity | DWA selections | `discovery_activity_selections` |
| Analysis | Scores, charts | `discovery_analysis_results` |
| Roadmap | Candidates, timeline | `agentification_candidates` |

---

## Appendix: Pew Research GWA AI Exposure Mapping

Base scores derived from Pew Research Center analysis of AI exposure by work activity type:

| GWA ID | Activity Name | AI Exposure |
|--------|---------------|-------------|
| 4.A.1.a.1 | Getting Information | 0.85 |
| 4.A.1.a.2 | Monitor Processes, Materials, or Surroundings | 0.70 |
| 4.A.1.b.1 | Identify Objects, Actions, and Events | 0.75 |
| 4.A.2.a.1 | Analyzing Data or Information | 0.90 |
| 4.A.2.a.2 | Making Decisions and Solving Problems | 0.65 |
| 4.A.2.a.3 | Thinking Creatively | 0.40 |
| 4.A.2.a.4 | Updating and Using Relevant Knowledge | 0.80 |
| 4.A.2.b.1 | Developing Objectives and Strategies | 0.55 |
| 4.A.2.b.2 | Scheduling Work and Activities | 0.75 |
| 4.A.3.a.1 | Performing General Physical Activities | 0.20 |
| 4.A.3.a.2 | Handling and Moving Objects | 0.35 |
| 4.A.3.a.3 | Controlling Machines and Processes | 0.50 |
| 4.A.3.a.4 | Operating Vehicles or Equipment | 0.45 |
| 4.A.3.b.1 | Interacting With Computers | 0.85 |
| 4.A.3.b.2 | Drafting, Laying Out, and Specifying | 0.70 |
| 4.A.3.b.4 | Repairing and Maintaining Equipment | 0.30 |
| 4.A.3.b.5 | Repairing and Maintaining Mechanical Equipment | 0.25 |
| 4.A.3.b.6 | Documenting/Recording Information | 0.90 |
| 4.A.4.a.1 | Interpreting Meaning of Information | 0.80 |
| 4.A.4.a.2 | Communicating with Supervisors or Peers | 0.55 |
| 4.A.4.a.3 | Communicating with Outside Organizations | 0.50 |
| 4.A.4.a.4 | Establishing and Maintaining Relationships | 0.30 |
| 4.A.4.a.5 | Assisting and Caring for Others | 0.25 |
| 4.A.4.a.6 | Selling or Influencing Others | 0.45 |
| 4.A.4.a.7 | Resolving Conflicts and Negotiating | 0.35 |
| 4.A.4.a.8 | Performing for or Working with Public | 0.30 |
| 4.A.4.b.1 | Coordinating Work of Others | 0.55 |
| 4.A.4.b.2 | Developing and Building Teams | 0.35 |
| 4.A.4.b.3 | Training and Teaching Others | 0.50 |
| 4.A.4.b.4 | Guiding, Directing, and Motivating | 0.35 |
| 4.A.4.b.5 | Coaching and Developing Others | 0.40 |
| 4.A.4.b.6 | Provide Consultation and Advice | 0.55 |
| 4.A.4.c.1 | Performing Administrative Activities | 0.80 |
| 4.A.4.c.2 | Staffing Organizational Units | 0.60 |
| 4.A.4.c.3 | Monitoring and Controlling Resources | 0.70 |

*Note: Scores are estimates and should be validated with domain experts.*

---

*Design completed: 2026-01-31*
*Brainstorming session with user*
