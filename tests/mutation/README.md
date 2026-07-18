# Mutation Testing with mutmut

Run mutation testing to validate test quality:

```bash
cd /path/to/cortex-vision
source .venv/bin/activate
mutmut run --paths-to-mutate src/
```

View results:
```bash
mutmut results
mutmut show <mutant-id>
```

Surviving mutants indicate gaps in the test suite that need additional tests.

**Note**: Mutation testing is slow (~5-30 min depending on codebase size).
