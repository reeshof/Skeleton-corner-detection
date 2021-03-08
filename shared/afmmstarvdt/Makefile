
.SUFFIXES:	.o .c .cpp .cc .C .a


RM = -rm


.cpp.o:	
	$(C++) $(CFLAGS) $(CCFLAGS) $(EXTRAFLAGS) $*.cpp

.c.o:
	$(CC)  $(CFLAGS) $(EXTRAFLAGS) $*.c


#############################################################

C++        = g++
CFLAGS     = -c -Iinclude 
CCFLAGS    = -fno-access-control -O2


VPATH   = Obj

CC_SRC   = flags.cpp io.cpp fmm.cpp genrl.cpp mfmm.cpp byteswap.cpp skeleton.cpp skeldt.cpp
OBJECTS  = flags.o io.o fmm.o genrl.o mfmm.o byteswap.o skeleton.o skeldt.o

default: skeleton

skeleton: $(OBJECTS) 
	-mv $(OBJECTS) Obj
	cd Obj ; $(C++) -o ../skeleton $(OBJECTS) -framework OpenGL -framework GLUT; cd .


clean:
	$(RM) $(VPATH)/*.o skeleton *~ OUTPUT/*
 

