include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
CFLAGS += -pedantic -std=c99

advect: advect.o
	-${CLINKER} -o advect advect.o  ${PETSC_LIB}
	${RM} advect.o

both: both.o
	-${CLINKER} -o both both.o  ${PETSC_LIB}
	${RM} both.o

# testing


.PHONY: clean distclean runadvect_1 runadvect_2 runadvect_3 runadvect_4 runboth_1 runboth_2 runboth_3 runboth_4 runboth_5 test_advect test_both test

distclean: clean

clean::
	@rm -f *~ *tmp *.pyc *.dat *.dat.info advect both
