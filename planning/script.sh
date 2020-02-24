# python Bagging_New_Cross_Blackbox.py -p AMWS -r 200
# python Bagging_New_Cross_Blackbox.py -p AMWS -r 500
#python Bagging_New_Cross_Blackbox.py -p AMWS -r 1000

# python Bagging_New_Cross_Blackbox.py -p mbe -r 200 -m dt
# python Bagging_New_Cross_Blackbox.py -p mbe -r 200 -m balance-dt
# python Bagging_New_Cross_Blackbox.py -p mbe -r 200 -m balance-svm
# python Bagging_New_Cross_Blackbox.py -p mbe -r 200 -m dummy
# python Bagging_New_Cross_Blackbox.py -p mbe -r 200 -m gp

# python Bagging_New_Cross_Blackbox.py -p mbe -r 500
# python Bagging_New_Cross_Blackbox.py -p mbe -r 1000

# python Bagging_New_Cross_Blackbox.py -p CRNP -r 200
# python Bagging_New_Cross_Blackbox.py -p CRNP -r 500
# python Bagging_New_Cross_Blackbox.py -p CRNP -r 1000

#python Bagging_New_Cross_Blackbox.py -p MFNP -r 1000 -m dummy
#python Bagging_New_Cross_Blackbox.py -p MFNP -r 1000 -m dt
#python Bagging_New_Cross_Blackbox.py -p MFNP -r 1000 -m balance-dt -static 1
#python Bagging_New_Cross_Blackbox.py -p MFNP -r 1000 -m balance-dt -simple 1 -static 1
#python Bagging_New_Cross_Blackbox.py -p MFNP -r 1000 -m svm
#python Bagging_New_Cross_Blackbox.py -p MFNP -r 1000 -m balance-svm -static 1
#python Bagging_New_Cross_Blackbox.py -p MFNP -r 1000 -m balance-svm -simple 1 -static 1
#python Bagging_New_Cross_Blackbox.py -p MFNP -r 1000 -m gp
#python Bagging_New_Cross_Blackbox.py -p MFNP -r 1000 -m balance-gp

# python Bagging_New_Cross_Blackbox.py -p Gonarezhou -r 1000 -m balance-dt
# python Bagging_New_Cross_Blackbox.py -p Gonarezhou -r 500

# python Bagging_New_Cross_Blackbox.py -p Gonarezhou_human -r 1000 -c human
# python Bagging_New_Cross_Blackbox.py -p MFNP -r 1000 -m balance-dt -simple 1 -cutoff 0 
# python Bagging_New_Cross_Blackbox.py -p MFNP -r 1000 -m balance-dt -simple 1 -cutoff 0.5
# python Bagging_New_Cross_Blackbox.py -p MFNP -r 1000 -m balance-dt -simple 1 -cutoff 1
# python Bagging_New_Cross_Blackbox.py -p MFNP -r 1000 -m balance-dt -simple 1 -cutoff 1.5
# python Bagging_New_Cross_Blackbox.py -p MFNP -r 1000 -m balance-dt -simple 1 -cutoff 2
# python Bagging_New_Cross_Blackbox.py -p MFNP -r 1000 -m balance-dt -simple 1 -cutoff 2.5
# python Bagging_New_Cross_Blackbox.py -p MFNP -r 1000 -m balance-dt -simple 1 -cutoff 3
# 
# python Bagging_New_Cross_Blackbox.py -p MFNP -r 1000 -m balance-svm -simple 1 -cutoff 0 
# python Bagging_New_Cross_Blackbox.py -p MFNP -r 1000 -m balance-svm -simple 1 -cutoff 0.5
# python Bagging_New_Cross_Blackbox.py -p MFNP -r 1000 -m balance-svm -simple 1 -cutoff 1
# python Bagging_New_Cross_Blackbox.py -p MFNP -r 1000 -m balance-svm -simple 1 -cutoff 1.5
# python Bagging_New_Cross_Blackbox.py -p MFNP -r 1000 -m balance-svm -simple 1 -cutoff 2
# python Bagging_New_Cross_Blackbox.py -p MFNP -r 1000 -m balance-svm -simple 1 -cutoff 2.5
# python Bagging_New_Cross_Blackbox.py -p MFNP -r 1000 -m balance-svm -simple 1 -cutoff 3


# ============================ 0820 experiment ================================
# cutoff 0
#python Bagging_New_Cross_Blackbox.py -p MFNP -r 1000 -m balance-dt -static 1
#python Bagging_New_Cross_Blackbox.py -p MFNP -r 1000 -m balance-dt -simple 1 -static 1
#python Bagging_New_Cross_Blackbox.py -p MFNP -r 1000 -m balance-svm -static 1
#python Bagging_New_Cross_Blackbox.py -p MFNP -r 1000 -m balance-svm -simple 1 -static 1

# cutoff 1
#python Bagging_New_Cross_Blackbox.py -p MFNP -r 1000 -m balance-dt -static 1 -cutoff 1
#python Bagging_New_Cross_Blackbox.py -p MFNP -r 1000 -m balance-dt -simple 1 -static 1 -cutoff 1
#python Bagging_New_Cross_Blackbox.py -p MFNP -r 1000 -m balance-svm -static 1 -cutoff 1
#python Bagging_New_Cross_Blackbox.py -p MFNP -r 1000 -m balance-svm -simple 1 -static 1 -cutoff 1

# cutoff 2
#python Bagging_New_Cross_Blackbox.py -p MFNP -r 1000 -m balance-dt -static 1 -cutoff 2
#python Bagging_New_Cross_Blackbox.py -p MFNP -r 1000 -m balance-dt -simple 1 -static 1 -cutoff 2
#python Bagging_New_Cross_Blackbox.py -p MFNP -r 1000 -m balance-svm -static 1 -cutoff 2
#python Bagging_New_Cross_Blackbox.py -p MFNP -r 1000 -m balance-svm -simple 1 -static 1 -cutoff 2


# ============================ 0821 experiment ================================
# python Bagging_New_Cross_Blackbox.py -p MFNP -r 1000 -m balance-svm -cutoff 0
# python Bagging_New_Cross_Blackbox.py -p MFNP -r 1000 -m balance-svm -cutoff 0.5
# python Bagging_New_Cross_Blackbox.py -p MFNP -r 1000 -m balance-svm -cutoff 1
# python Bagging_New_Cross_Blackbox.py -p MFNP -r 1000 -m balance-svm -cutoff 1.5
# python Bagging_New_Cross_Blackbox.py -p MFNP -r 1000 -m balance-svm -cutoff 2
# python Bagging_New_Cross_Blackbox.py -p MFNP -r 1000 -m balance-svm -cutoff 2.5
# python Bagging_New_Cross_Blackbox.py -p MFNP -r 1000 -m balance-svm -cutoff 3
# python Bagging_New_Cross_Blackbox.py -p MFNP -r 1000 -m balance-svm -cutoff 3.5
# python Bagging_New_Cross_Blackbox.py -p MFNP -r 1000 -m balance-svm -cutoff 4

