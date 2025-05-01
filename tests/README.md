# Tests

This directory contains unit tests for the Hopfield Network Market Regime Classifier.

## Running Tests

You can run all the tests with the test runner script:

```bash
python run_tests.py
```

Or run individual tests:

```bash
python -m unittest tests/test_hopfield_network.py
```

## Test Coverage

The tests cover the following components:

- **Hopfield Network**: Core functionality of the Hopfield Network implementation including pattern storage, neuron updates, network dynamics, energy calculation, and pattern classification.

More tests will be added for:

- Data acquisition module
- Feature engineering module
- Market regime classifier module

## Writing Tests

When adding new tests, follow these guidelines:

1. Create a new test file with the prefix `test_` for the module you want to test.
2. Use the `unittest` framework.
3. Organize tests in classes inheriting from `unittest.TestCase`.
4. Name test methods with the prefix `test_`.
5. Include docstrings describing what each test is checking.
6. Use assertions to verify expected behavior. 