set.seed(2017)

#a
samples = rnorm(n=5, mean=0, sd=1)

#b
mean_ml = mean(samples)
var_ml = var(samples)*((5-1)/5)

print('samples are:')
print(samples) #1.43420148 -0.07729196  0.73913723 -1.75860473 -0.06982523
print('MLE Mean:')
mean_ml #0.05352336
print('MLE Var:')
var_ml #1.138495

#c
mean_ml_vec = rep(0, 10000)
var_ml_vec =rep(0, 10000)

for (i in 1:10000){
  samples = rnorm(n=5, mean=0, sd=1)
  mean_ml = mean(samples)
  var_ml = var(samples)*((5-1)/5)
  
  mean_ml_vec[i] = mean_ml
  var_ml_vec[i] = var_ml
}
print("Computed 10000 samples, generating plot")
hist(var_ml_vec, xlab="MLE Var", ylab="P(MLE Var)", main="Empirical Distribution of MLE Variances",breaks=20)
print("Plot generated. Computing Bias and Variance")

#d
print("Estimated Bias is:")
mean(var_ml_vec)-1 #-0.2013675
#Variance
print("Estimated Variance is:")
var(var_ml_vec) #0.3188041
