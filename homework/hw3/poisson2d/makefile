include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
CFLAGS += -pedantic -std=c99

poisson: poisson.o  
	-${CLINKER} -o poisson poisson.o ${PETSC_LIB}
	${RM} poisson.o

# etc

.PHONY: clean distclean 

distclean: clean

clean::
	@rm -f *~ poisson *tmp *.vtr

