# TODO / Follow-ups

## Completed ✅
- [x] Create config/assets.yaml for symbol classification
- [x] Enhance ModelRouter with asset classification from config  
- [x] Create crypto model training script
- [x] Run live dry-run validation with crypto model
- [x] Create data contracts and determinism infrastructure
- [x] Fix pandas assert_frame_equal usage and harden determinism checks
- [x] Add ONNX export and parity testing infrastructure
- [x] Add IC/hit-rate metrics for crypto evaluation
- [x] Create golden crypto snapshot for CI
- [x] Add crypto smoke and parity jobs to CI
- [x] Fix asset classification fallback to use 'equities' instead of 'universal'

## Remaining Tasks
- [ ] Set up live metrics collection and artifacts (pending live deployment)
- [ ] Prepare equities model for Monday (pending equities model training)

## Notes
- All crypto infrastructure is now complete and tested
- CI jobs are configured and passing
- Asset routing is working correctly with proper fallbacks
- Ready for live deployment when needed

(Links: see ../15-30-crypto-ci-jobs/)

