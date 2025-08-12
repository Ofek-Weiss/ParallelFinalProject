TARGET1 = parallel
SRC1 = main.c

TARGET2 = serial
SRC2 = serial.c

CC = gcc
MPICC = mpicc

all:
	$(MPICC) -fopenmp -o $(TARGET1) $(SRC1) -lm
	$(CC) -o $(TARGET2) $(SRC2) -lm

clean:
	rm -f $(TARGET1) $(TARGET2)

