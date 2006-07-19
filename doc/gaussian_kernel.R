




gaussian <- function(u, v) {
  return ( exp( - (u-v)^2 / (2*1^2) ) )
}

N <- 50
x <- seq(-3,3,length=N)


plot( x, gaussian(x,0), type="l", xlab="u", ylab="k(u,0)" )

