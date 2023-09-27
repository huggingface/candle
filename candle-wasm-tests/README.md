Run the tests with:
```bash
RUST_LOG=wasm_bindgen_test_runner wasm-pack test --chrome --headless
```
Or:
```bash
wasm-pack test --chrome
```

If you get an "invalid session id" failure in headless mode, check that logs and
it may well be that your ChromeDriver is not at the same version as your
browser.
