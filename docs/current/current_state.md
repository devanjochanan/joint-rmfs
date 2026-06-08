# Rika RMFS Current State

This document captures the current branch details, phase progress, and simulation source of truth.

---

## 1. Git Environment & Preflight Summary
* **Current Branch**: `main`
* **Remote origin**: `https://github.com/devanjochanan/joint-rmfs.git`
* **Status**: Clean baseline branch. Only local ignored CSV files are modified in workspace but ignored from tracking.

---

## 2. Refactoring Phases Progress

### Phase 1: Repository Inventory (Completed)
* **Goal**: Document the existing file structures, components, and identify critical files that contain core behavior logic.
* **Output**: Created `docs/architecture/file_inventory.md` and `docs/architecture/current_architecture_map.md` detailing the file listings and risk profiles.

### Phase 1.5: Artifact Hygiene (Completed)
* **Goal**: Clean up Git tracking for runtime, local, and generated files (databases, logs, local settings) to avoid noise and merge conflicts.
* **Output**: Modified `.gitignore` and removed target files/folders from git index cache (using `git rm --cached`). Staged and committed deletions of:
  * Local metadata (`.DS_Store`, `.claude/`, `.vscode/`)
  * Runtime DBs and states (`warehouse.db`, `warehouse_ps_old_8.db`, `netlogo.state`, `profile.prof`)
  * Dynamic tracking CSVs (`assign_order.csv`, `pod_info.csv`)
  * Run-specific data and outputs (`PS/`, `output/`, `robot sa data/`)

### Phase 2: Package Scaffold & Ownership Definition (In Progress)
* **Goal**: Set up a modular folder skeleton under `src/rmfs/` and create ownership profiles for the collaborating researchers.
* **Output**: Folder scaffold created under `src/rmfs/` with README files explaining package roles. Ownership profiles created under `docs/modules/`.

---

## 3. Core Refactoring Policies

> [!IMPORTANT]
> **Behavior Preservation Baseline Rule**
> The primary constraint of this refactoring process is to **fully preserve the current simulation behavior**.
> 
> * **No Feature Imports**: No new algorithms, regrets, or deep learning models are imported in this phase.
> * **No Behavior Refactoring**: No logic is shifted or rewritten.
> * **Behavior Source of Truth**: The active source files at the root directory—namely `simulation.nlogo`, `netlogo.py`, `engine/**`, and `model/**`—remain the sole, unmodified source of truth for the simulation behavior at this stage. All scaffold directories under `src/rmfs/` are strictly placeholders and must not be imported or called yet.
