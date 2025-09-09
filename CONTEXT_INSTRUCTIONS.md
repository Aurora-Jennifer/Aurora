# Context Management Instructions

## 🎯 How to Use This System

### Before Starting Any Work
1. **Always read `MASTER_CONTEXT.md` first**
2. **Check the file status** (✅❌🔄) before editing
3. **Don't rewrite files marked as ✅ WORKING**
4. **Focus on files marked as ❌ BROKEN or 🔄 IN PROGRESS**

### When Making Changes
1. **Update `MASTER_CONTEXT.md`** with:
   - File status changes
   - New test results
   - Resolved blockers
   - Updated next actions

### Quick Commands for Me

#### To Check Current Status
```
"Check MASTER_CONTEXT.md and tell me current status"
```

#### To Focus on Specific Issue
```
"Focus on [ISSUE_NAME] from MASTER_CONTEXT.md blockers"
```

#### To Update Context
```
"Update MASTER_CONTEXT.md with [CHANGE_DESCRIPTION]"
```

#### To Avoid Rewriting
```
"Check MASTER_CONTEXT.md before making changes to [FILE_NAME]"
```

## 🚨 Red Flags to Watch For

### Don't Rewrite These Files
- Files marked as ✅ WORKING
- Files marked as ❌ EXISTS, NOT USED
- Files that already solve the problem

### Always Check First
- Does the file already exist?
- What's its current status?
- Is there a better existing solution?

## 📋 Current Priority (from MASTER_CONTEXT.md)

### CRITICAL: Fix Model Interface
- **File**: `core/ml/advanced_models.py`
- **Issue**: EnsembleRewardModel not callable
- **Action**: Add `predict()` method

### HIGH: Fix Training Pipeline
- **File**: TrainingConfig loading
- **Issue**: Missing required arguments
- **Action**: Fix config loading from metadata

### MEDIUM: Re-run Validation
- **File**: `scripts/validate_model_walkforward.py`
- **Action**: Test with fixed model interface

## 🔄 Context Update Template

When updating MASTER_CONTEXT.md, use this format:

```markdown
### [File Name]
- [Status Change]: [Description]
- **Result**: [What happened]
- **Next**: [What to do next]
```

## 📊 Status Legend
- ✅ **WORKING** - File works as intended
- ❌ **BROKEN** - File has issues, needs fixing
- 🔄 **IN PROGRESS** - Currently being worked on
- ❌ **EXISTS, NOT USED** - File exists but not relevant to current mission
