LIBS+= -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_legacy -lopencv_calib3d -lopencv_features2d
LIBS+= -lglut -lGL -lGLU -lGLEW
LIBS+= -lmrpt-base -lmrpt-slam -lmrpt-obs -lmrpt-vision -lmrpt-opengl -lmrpt-maps -lmrpt-bayes -lmrpt-gui -lmrpt-scanmatching
LIBS+= -lcsparse -luuid -llcm
LIBS+= -lX11 -lz
#LIBS+= /usr/include/zlib.h /usr/include/X11/X.h
#-lcoredumper


LIB_DIRS+= -L/home/calum/fuerte_workspace/sandbox/mapbuilder/include

INCLUDES += -I/usr/local/include/lcm
INCLUDES += -I/home/calum/fuerte_workspace/sandbox/mapbuilder/include
#INCLUDES += -isystem /usr/include/opencv-2.3.1 -isystem /usr/include/opencv-2.3.1/opencv
INCLUDES += -isystem /usr/local/include/opencv2 -isystem /usr/local/include/opencv
INCLUDES += -isystem /usr/include/mrpt/mrpt-config
INCLUDES += -isystem /usr/include/mrpt/base/include
INCLUDES += -isystem /usr/include/mrpt/slam/include
INCLUDES += -isystem /usr/include/mrpt/obs/include
INCLUDES += -isystem /usr/include/mrpt/vision/include
INCLUDES += -isystem /usr/include/mrpt/opengl/include
INCLUDES += -isystem /usr/include/mrpt/maps/include
INCLUDES += -isystem /usr/include/mrpt/bayes/include
INCLUDES += -isystem /usr/include/mrpt/gui/include
#INCLUDES += -isystem /home/calum/mapbuilder/include
INCLUDES += -isystem /usr/include/X11

CPP_FLAGS += -O2  -fPIC -Wcomment -Wno-unused-but-set-variable -Wno-write-strings
CPP_FLAGS += -ggdb
# -O3 -msse2 -Wall

TARGETS: mapbuilder

all : TARGETS



%.o: %.cpp
	g++ $^ $(FLAGS) $(LIBS) $(INCLUDES) $(LIB_DIRS) $(CPP_FLAGS) -o $@ -c

%.o: %.cc
	g++ $^ $(FLAGS) $(LIBS) $(INCLUDES) $(LIB_DIRS) $(CPP_FLAGS) -o $@ -c

%.o: %.c
	g++ $^ $(FLAGS) $(LIBS) $(INCLUDES) $(LIB_DIRS) $(CPP_FLAGS) -o $@ -c

clean:
	rm -rf *.o *~

OBJECTS += Utm.o Mapping.o MapBuilder.o simplelog.o spa2d.o csparse.o

mapbuilder: main.o $(OBJECTS)
	g++ $^ $(FLAGS) $(LIBS) $(INCLUDES) $(LIB_DIRS) $(CPP_FLAGS) -o ../$@
