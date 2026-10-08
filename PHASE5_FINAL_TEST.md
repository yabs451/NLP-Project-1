# Phase 5 final-test results

The frozen models were evaluated once on the reserved final-test set
after all experimental choices had been fixed.

Main findings:

- Base recursion remains stable: g4 test accuracy = 0.997.
- Label argmax remains stable: g4 = 0.996.
- Label sampling T=3 collapses: g4 = 0.497.
- Label sampling T=5 collapses more strongly: g4 = 0.224.
- Context feedback T=1/3 shows late collapse: g6 = 0.403.
- Next-symbol T=0.2 retains high query accuracy: g6 = 0.991,
  while following-label accuracy falls to 0.937 and symbol loss reaches 5.837.

These results reproduce the qualitative development-set findings on
the untouched final-test classes.

No model selection, retraining, hyperparameter tuning, metric changes
or new experiments were performed after observing final-test results.
