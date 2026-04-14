include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
CFLAGS += -pedantic -std=c99 


ts_advection_diffusion: ts_advection_diffusion.o  
	-${CLINKER} -o ts_advection_diffusion ts_advection_diffusion.o ${PETSC_LIB}
	${RM} ts_advection_diffusion.o

#
# etc			

.PHONY: clean distclean 
distclean: clean

clean::
	@rm -f *~ ts_advection_diffusion *tmp

