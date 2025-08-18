# 📚 Context Files for AI Prompting

## **Purpose**
These context files provide comprehensive information about the trading system codebase to enable better AI assistance and more accurate responses to development questions.

## **Context Files Overview**

### **1. System Overview** (`01_SYSTEM_OVERVIEW.md`)
**Use for**: General understanding of the system, architecture, and current status
- System identity and capabilities
- Core architecture and components
- Current system status (working/broken components)
- Performance metrics and configuration system
- Development philosophy and next priorities

### **2. Critical Issues** (`02_CRITICAL_ISSUES.md`)
**Use for**: Immediate problem-solving and urgent fixes
- 5 critical errors requiring immediate attention
- Detailed problem descriptions and solutions
- Testing strategies for each fix
- Priority matrix and success criteria
- Deployment checklist

### **3. Development Philosophy** (`03_DEVELOPMENT_PHILOSOPHY.md`)
**Use for**: Coding standards, best practices, and architectural decisions
- Core development principles (no hardcoded values, safety first, etc.)
- Code quality standards and patterns
- Configuration management approach
- Error handling patterns and testing standards
- Known hotspots and maintenance guidelines

### **4. System Architecture** (`04_SYSTEM_ARCHITECTURE.md`)
**Use for**: Understanding codebase structure and component relationships
- Complete project layout and file organization
- Core components architecture and responsibilities
- Data flow and integration points
- Performance and testing architecture
- Deployment architecture

### **5. Next Session Plan** (`05_NEXT_SESSION_PLAN.md`)
**Use for**: Planning immediate development work and priorities
- Critical tasks and timeline
- Detailed fix instructions for each issue
- Testing strategies and success metrics
- Risk mitigation and contingency plans
- Session preparation checklist

## **How to Use These Context Files**

### **For General Questions**
Start with `01_SYSTEM_OVERVIEW.md` to understand the system context, then reference specific files as needed.

### **For Bug Fixes**
1. Check `02_CRITICAL_ISSUES.md` for known issues and solutions
2. Reference `03_DEVELOPMENT_PHILOSOPHY.md` for coding standards
3. Use `04_SYSTEM_ARCHITECTURE.md` to understand component relationships

### **For New Features**
1. Review `01_SYSTEM_OVERVIEW.md` for system capabilities
2. Check `03_DEVELOPMENT_PHILOSOPHY.md` for development standards
3. Use `04_SYSTEM_ARCHITECTURE.md` to understand where to add code

### **For Planning Sessions**
1. Start with `05_NEXT_SESSION_PLAN.md` for immediate priorities
2. Reference `02_CRITICAL_ISSUES.md` for urgent fixes
3. Use `03_DEVELOPMENT_PHILOSOPHY.md` for implementation guidelines

### **For Code Reviews**
1. Use `03_DEVELOPMENT_PHILOSOPHY.md` for coding standards
2. Reference `04_SYSTEM_ARCHITECTURE.md` for architectural consistency
3. Check `02_CRITICAL_ISSUES.md` for common pitfalls

## **Context File Relationships**

```
01_SYSTEM_OVERVIEW.md
├── High-level system understanding
├── Current status and capabilities
└── Links to specific areas

02_CRITICAL_ISSUES.md
├── Immediate problems to solve
├── Detailed solutions and testing
└── Priority and impact assessment

03_DEVELOPMENT_PHILOSOPHY.md
├── Coding standards and principles
├── Best practices and patterns
└── Quality assurance guidelines

04_SYSTEM_ARCHITECTURE.md
├── Codebase structure and organization
├── Component relationships and data flow
└── Integration points and interfaces

05_NEXT_SESSION_PLAN.md
├── Immediate development priorities
├── Detailed task breakdown
└── Success criteria and timeline
```

## **Prompting Examples**

### **Example 1: Fix a Bug**
```
Context: I'm working on the trading system described in the context files.
I need to fix a memory leak in the composer integration (see 02_CRITICAL_ISSUES.md).
The issue is in core/engine/composer_integration.py where copy.deepcopy() is used.
Please help me implement the solution following the development philosophy in 03_DEVELOPMENT_PHILOSOPHY.md.
```

### **Example 2: Add a New Feature**
```
Context: I'm adding a new technical indicator to the trading system (see 01_SYSTEM_OVERVIEW.md).
I need to understand where to place it in the architecture (see 04_SYSTEM_ARCHITECTURE.md)
and follow the development standards (see 03_DEVELOPMENT_PHILOSOPHY.md).
Please help me implement this feature.
```

### **Example 3: Plan Development Session**
```
Context: I'm planning the next development session for the trading system.
Please review the next session plan (05_NEXT_SESSION_PLAN.md) and critical issues (02_CRITICAL_ISSUES.md)
to help me prioritize the work and estimate timelines.
```

### **Example 4: Code Review**
```
Context: I'm reviewing code changes for the trading system.
Please help me check if the changes follow the development philosophy (03_DEVELOPMENT_PHILOSOPHY.md)
and are consistent with the system architecture (04_SYSTEM_ARCHITECTURE.md).
```

## **Context File Maintenance**

### **When to Update**
- **After major changes**: Update relevant context files
- **New critical issues**: Add to `02_CRITICAL_ISSUES.md`
- **Architecture changes**: Update `04_SYSTEM_ARCHITECTURE.md`
- **New development priorities**: Update `05_NEXT_SESSION_PLAN.md`

### **Update Process**
1. Identify which context files are affected
2. Update the specific sections
3. Ensure consistency across all files
4. Update this README if file relationships change

### **Version Control**
- Keep context files in version control
- Update them as part of the development process
- Use them for onboarding new team members
- Reference them in code reviews and planning sessions

## **Benefits of Using Context Files**

### **For AI Assistance**
- **Better Understanding**: AI has comprehensive context about the system
- **Accurate Responses**: Responses are tailored to the specific codebase
- **Consistent Guidance**: Follows established patterns and standards
- **Efficient Problem Solving**: Quick access to known issues and solutions

### **For Development**
- **Faster Onboarding**: New developers can quickly understand the system
- **Consistent Standards**: All development follows established patterns
- **Better Planning**: Clear priorities and timelines for development
- **Reduced Errors**: Awareness of common pitfalls and solutions

### **For Maintenance**
- **Documentation**: Comprehensive system documentation
- **Troubleshooting**: Quick access to known issues and fixes
- **Architecture**: Clear understanding of system structure
- **Standards**: Consistent development practices

---

**Last Updated**: August 17, 2025
**Purpose**: Enable better AI assistance and development efficiency
**Maintenance**: Update as system evolves
