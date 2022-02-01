# Motility-emergence-morpho-assymetry
### Author: Joan Térmens

This respository contains the attempts to solve the system of polarity and force balance quations in order to find the velocity field.
The [Deprecated folder](./Deprecated) contains previus attemps to solve the problem with
[fenics](./Deprecated/fenics) and [matlab](./Deprecated/matlab), as well as all the reference codes from Ido Lavi, stored at the [freefem++ folder](./Deprecated/freefem++)

The different algorithms start by solving the polarity equation and then try to solve the force balance equation to find the velocity using the previous solution
for the polarity. The polarity equation is correctly solved in both the python and matlab scripts, whereas the force balance equation presents a series of problems
that result in pathological solutions.
