# Identification Data

Place your dynamic identification data in this directory.

## Expected Format

CSV files with columns:
- timestamp
- positions (q1, q2, q3, ...)
- velocities (dq1, dq2, dq3, ...)
- accelerations (ddq1, ddq2, ddq3, ...) [optional, can be computed]
- torques (tau1, tau2, tau3, ...)

Example:
```
timestamp,q1,q2,q3,dq1,dq2,dq3,tau1,tau2,tau3
0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
0.01,0.001,0.002,0.001,0.1,0.2,0.1,1.5,2.3,0.8
...
```
