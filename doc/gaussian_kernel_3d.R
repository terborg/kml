




gaussian <- function(u, v) {
  return ( exp( - t(u-v) %*% (u-v) / (2*1^2) ) )
}


N <- 25
x <- seq(-3,3,length=N)

y <- matrix( nrow=N, ncol=N )
for( i in 1:N ) {
  for( j in 1:N ) {
    y[i,j] <- gaussian( c(x[i],x[j]), c(0,0) )
  }
}


#par("mfcol"=c(1,2))
# 3d version Gaussian kernel
persp(x,x,y, col="lightblue", phi=40, theta=120, shade=0.3 )






