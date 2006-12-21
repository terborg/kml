


N <- 50
x <- vector(length=N)
for( i in 1:N ) {
 x[i] <- (i-1) / (N-1) * 20.0 - 10.0
}
y <- sin(x) / x + rnorm( N, 0, 0.1 )


halfplot()
plot( x, y, xlab="x", ylab="sinc(x)+N(0,0.1)" )

