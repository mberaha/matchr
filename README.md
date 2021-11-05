# matchr
Optimal transportation of discrete measures and matching of point clouds in R

Example usage

```
devtools::install_github("https://github.com/mberaha/matchr")
x = matrix(rnorm(6), nrow=3, ncol=2)
y = matrix(rnorm(6), nrow=3, ncol=2)
w = c(1, 1, 1) / 3
tranport_plan(x, w, y, w)

m = match(x, y)

plot(x[, 1], x[, 2],  pch=16, xlim = c(-3, 3), ylim=c(-3, 3))
points(y[, 1], y[, 2],  pch=16, col="red")
for (i in 1:3) {
  lines(c(x[i, 1], y[m[i], 1]), c(x[i, 2], y[m[i], 2]))
}
```



