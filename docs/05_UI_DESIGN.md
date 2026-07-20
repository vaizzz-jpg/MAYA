# MAYA — UI Design Planning (No Implementation)

## 1. Why this exists

MAYA must look and feel like **investigation software**, not a marketing landing site or ML demo. This document defines screens, purpose, and components before coding UI beyond the Phase 1 shell.

**Design principles**

- Case-first navigation  
- Dense but calm workspace (tables > marketing cards)  
- Evidence integrity visible (hash, timestamps)  
- Analysis explanations adjacent to scores  
- Bootstrap 5 components first; custom CSS only when necessary  
- Chart.js for charts; AOS.js for light section motion on marketing/landing only  

---

## 2. Global chrome

### Top navigation (authenticated)

- Brand: **MAYA**  
- Links: Dashboard | Cases | Evidence (optional shortcut) | Reports | Settings  
- Right: user menu (profile / logout)  

### Top navigation (anonymous / Phase 1)

- Brand + Login (placeholder link until auth) + Health/status subtle indicator optional  

### Footer

- Product name, version, “Authorized use only” notice  

---

## 3. Screen catalog

### 3.1 Landing

**Purpose:** Introduce MAYA as a forensic investigation platform and route authorized users to login.

**Components**

- Hero with product name **MAYA** as dominant brand signal  
- One short headline + one supporting sentence  
- Primary CTA: Login / Enter workspace  
- Secondary: documentation link (optional)  

**Buttons:** Login, Learn more (scroll)  
**Cards:** Avoid in hero; optional feature strip below fold only  
**Tables:** None  
**Navigation:** Minimal public nav  

---

### 3.2 Login

**Purpose:** Authenticate investigators.

**Components**

- Centered form: username/email, password, remember-me (optional)  
- Error alert for invalid credentials  
- Link to admin contact (no public self-registration by default)  

**Buttons:** Sign in  
**Cards:** Optional single form panel (interaction container—acceptable)  
**Tables:** None  
**Navigation:** Brand home + login only  

---

### 3.3 Dashboard

**Purpose:** Operational overview for the investigator’s workload.

**Components**

- Summary metrics (open cases, evidence pending analysis, recent reports) — below any header, using Bootstrap rows  
- Recent cases table  
- Recent analysis activity list  
- Chart.js small charts: case status distribution / analyses this week  

**Buttons:** New Case, Upload Evidence  
**Cards:** Metric widgets allowed as interaction/overview containers  
**Tables:** Recent cases  
**Navigation:** Full authenticated nav; Dashboard active  

---

### 3.4 Case Management

**Purpose:** Create, search, and manage investigation cases.

**List view components**

- Search + status filter  
- Cases table: number, title, status, evidence count, updated_at  
- Pagination  

**Create/Edit components**

- Form: case number (auto or manual), title, description, status  

**Buttons:** New Case, Open, Edit, Close/Archive  
**Cards:** Prefer table-first; form panel for create  
**Tables:** Primary content  
**Navigation:** Cases active  

---

### 3.5 Evidence Upload

**Purpose:** Attach digital evidence to a case with integrity recording.

**Components**

- Case selector / locked case context breadcrumb  
- File input (drag-drop if Bootstrap-friendly custom minimal JS)  
- Allowed types / max size hints  
- Optional investigator notes  
- Post-upload confirmation showing **SHA-256**  

**Buttons:** Upload, Cancel  
**Cards:** Upload panel  
**Tables:** None (or recent uploads mini-table)  
**Navigation:** From case detail  

---

### 3.6 Evidence Viewer

**Purpose:** Inspect an evidence item without mutating originals.

**Components**

- Media preview (image)  
- Integrity panel: hash, size, mime, uploaded_by, timestamps  
- Linked case breadcrumb  
- Notes  
- List of prior analysis runs  

**Buttons:** Run Analysis, Generate Report (if eligible), Download original (policy-gated)  
**Cards:** Metadata panel  
**Tables:** Analysis history  
**Navigation:** Case → Evidence  

---

### 3.7 Analysis Screen

**Purpose:** Show model outcomes and explanations for decision support—not a party trick.

**Components**

- Prediction label + confidence / probability  
- Authenticity score (Chart.js gauge or bar)  
- Forensic/metadata signal list (separate from AI score)  
- Grad-CAM / explanation image beside original  
- Model name + version (reproducibility)  
- Status timeline of the run  

**Buttons:** Re-run analysis, Save notes, Generate report  
**Cards:** Score panel + explanation panel  
**Tables:** Optional feature contribution table if available  
**Navigation:** Evidence → Analysis  

---

### 3.8 Report Screen

**Purpose:** Compose and download investigation PDFs.

**Components**

- Case summary  
- Selected evidence / analyses checklist  
- Preview metadata of last generated report  
- Download link  

**Buttons:** Generate PDF, Download  
**Cards:** Report configuration panel  
**Tables:** Generated reports history  
**Navigation:** Reports or Case → Report  

---

### 3.9 Settings

**Purpose:** Operational configuration for admins (and limited prefs for investigators).

**Components**

- Upload size limits  
- Allowed MIME list  
- Retention notes  
- Account password change (investigator)  

**Buttons:** Save settings  
**Cards:** Settings sections  
**Tables:** Optional settings key list for admins  
**Navigation:** Settings active; admin-gated sections  

---

## 4. Error pages (Phase 1 includes shell)

| Page | Purpose |
|------|---------|
| 404 | Resource not found — calm professional message |
| 403 | Forbidden — no privilege disclosure beyond necessity |
| 500 | Server error — log ID only in future; generic message now |

---

## 5. Phase 1 UI deliverable (only)

- Base layout with Bootstrap 5 + Bootstrap Icons CDN  
- Navigation placeholder  
- Simple home/foundation page  
- Error templates wired  
- No case/auth business screens yet  

Full screens above are **designed now**, implemented in later phases.
