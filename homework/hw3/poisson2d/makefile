include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
CFLAGS += -pedantic -std=c99

poisson2d: poisson2d.o  
	-${CLINKER} -o poisson2d poisson2d.o ${PETSC_LIB}
	${RM} poisson2d.o

# etc

.PHONY: clean distclean 

distclean: clean

clean::
	@rm -f *~ poisson2d *tmp *.vtr

