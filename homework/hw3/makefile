include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
CFLAGS += -pedantic -std=c99

reaction2d: reaction2d.o  
	-${CLINKER} -o reaction2d reaction2d.o ${PETSC_LIB}
	${RM} reaction2d.o

# etc

.PHONY: clean distclean 

distclean: clean

clean::
	@rm -f *~ reaction2d *tmp *.vtr

