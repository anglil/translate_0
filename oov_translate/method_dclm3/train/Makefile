cc=g++
LIBS=-L/home/ec2-user/kklab/src/dynet/build/dynet -ldynet -L/home/ec2-user/kklab/src/boost_1_61_0/lib -lboost_serialization -lboost_filesystem -lboost_system -lboost_program_options -lstdc++ -lm
#LIBS=-L/home/ec2-user/kklab/src/dynet/build/dynet -lgdynet -L/home/ec2-user/kklab/src/boost_1_61_0/lib -lboost_serialization -lboost_filesystem -lboost_system -lboost_program_options -lstdc++ -lm
CFLAGS=-I/home/ec2-user/kklab/src/dynet -I/home/ec2-user/kklab/src/eigen -I/home/ec2-user/kklab/src/dynet/external/easyloggingpp/src -I/home/ec2-user/kklab/src/boost_1_61_0 -std=gnu++11 -g

OBJ=util.o dclm.o charlm.o latticedec0.o latticedec1.o 
#latticedec2.o latticedec3.o

all: dclm charlm latticedec0 latticedec1 
#latticedec2 latticedec3

%.o: %.cc
	$(CC) $(CFLAGS) -c -o $@ $<

dclm: dclm.o util.o
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)
charlm: charlm.o util.o
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)
latticedec0: latticedec0.o util.o
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)
latticedec1: latticedec1.o util.o
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)
#latticedec2: latticedec2.o util.o
#	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)
#latticedec3: latticedec3.o util.o
#	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

clean:
	rm -rf *.o *.*~ dclm charlm latticedec0 latticedec1 
#latticedec2 latticedec3
