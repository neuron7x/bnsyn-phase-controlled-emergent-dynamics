# Deferred Gates

## actionlint (workflow lint)

**Status**: DEFERRED  
**Reason**: `actionlint` is not available in the current environment (`command not found`).  
**Reproduction**:
```bash
actionlint -verbose
```
**Next step**: Add a tooling PR to install `actionlint` in the developer/CI environment
and run it as part of workflow validation.
