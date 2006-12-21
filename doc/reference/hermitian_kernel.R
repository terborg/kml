

gaussian <- function(x, s) {
  return ( exp( -x^2 / (2*s^2) ) )
}

gaussianD1 <- function(x,s) {
  u <- sqrt(2)*s
  v <- 1/u
  return ( (-u)^-1 * (2*v*x) * exp( -x^2 / (2*s^2) ) )
}

gaussianD2 <- function(x,s) {
  u <- sqrt(2)*s
  v <- 1/u
  return ( (-u)^-2 * (4*(v*x)^2-2) * exp( -x^2 / (2*s^2) ) )
}

gaussianD3 <- function(x,s) {
  u <- sqrt(2)*s
  v <- 1/u
  return ( (-u)^-3 * (8*(v*x)^3-12*(v*x)) * exp( -x^2 / (2*s^2) ) )
}

gaussianD4 <- function(x,s) {
  u <- sqrt(2)*s
  v <- 1/u
  return ( (-u)^-4 * (16*(v*x)^4-48*(v*x)^2+12) * exp( -x^2 / (2*s^2)) )
}

gaussianD5 <- function(x,s) {
  u <- sqrt(2)*s
  v <- 1/u
  return ( (-u)^-5 * (32*(v*x)^5-160*(v*x)^3+120*(v*x)) * exp( -x^2 / (2*s^2) ) )
}


fullplot()
par("mfcol"=c(2,3))
x <- seq(-3,3,length=100)
plot( x, gaussian(x,1), type="l", ylab=expression(k(bold(x),bold(x)[i])) )
plot( x, gaussianD3(x,1), type="l", ylab=expression(D^3*k(bold(x),bold(x)[i])) )
plot( x, gaussianD1(x,1), type="l", ylab=expression(D^1*k(bold(x),bold(x)[i])) )
plot( x, gaussianD4(x,1), type="l", ylab=expression(D^4*k(bold(x),bold(x)[i])) )
plot( x, gaussianD2(x,1), type="l", ylab=expression(D^2*k(bold(x),bold(x)[i])) )
plot( x, gaussianD5(x,1), type="l", ylab=expression(D^5*k(bold(x),bold(x)[i])) )




