# to add new objects, just put them in this list and run make depend

OBJ = bid.o BidSet.o Distribution.o arbitrary.o Legacy.o main.o matching.o normal.o Param.o paths.o regions.o scheduling.o polyModel.o featureCalc.o



########################################################
# UNCOMMENT/EDIT THESE TO COMPILE WITH CPLEX LIBRARIES
# ########################################################
# (Comment everything in this section out)
# INCLUDE = ...
# LIB = ...
# LIBDIRS = ...

# ########################################################
# # UNCOMMENT/EDIT THESE TO COMPILE WITH LPSOLVE
# ########################################################
INCLUDE = -Ilp_solve_4.0
LIB = -lm -llpk
LIBDIRS = -Llp_solve_4.0

RELEASEFLAGS = -O5 -DNDEBUG
DEBUGFLAGS = -Wall -DDEBUG -g
LPMAKE = cd lp_solve_4.0; make;
LPCLEAN = cd lp_solve_4.0; make clean;
############################################################
release:
	cd obj; make all "OBJ = ${OBJ}" "CPPFLAGS = ${RELEASEFLAGS}";
	${LPMAKE}
	make all "VPATH = obj" "FLAGS = -o cats ${RELEASEFLAGS}"

debug:
	cd debugobj; make all "OBJ = ${OBJ}" "CPPFLAGS = ${DEBUGFLAGS}";
	${LPMAKE}
	make all "VPATH = debugobj" "FLAGS = -o debugcats ${DEBUGFLAGS}"

all : $(OBJ)
	g++ $^ ${FLAGS} ${LIBDIRS} $(LIB)

.PHONY: clean depend again

clean:
	rm *obj/*.o ;

depend:
	makedepend -Y -fobj/Makefile *.cc *.cpp # >& /dev/null
	makedepend -Y -fdebugobj/Makefile *.cc *.cpp # >& /dev/null

again:
	make clean; make depend; make
# DO NOT DELETE
