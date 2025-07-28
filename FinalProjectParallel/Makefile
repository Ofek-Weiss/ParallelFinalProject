TARGET1 = parallel
SRC1 = main.c

TARGET2 = serial
SRC2 = serial.c

CFLAGS = -fopenmp
LDFLAGS = -lm
MPICC = mpicc
CC = gcc

all:
	$(MPICC) $(CFLAGS) -o $(TARGET1) $(SRC1) $(LDFLAGS)
	$(CC) -o $(TARGET2) $(SRC2) $(LDFLAGS)

clean:
	rm -f $(TARGET1) $(TARGET2)

