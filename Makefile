EXEC = \
    tests/test-matrix \
    tests/test-stopwatch

GIT_HOOKS := .git/hooks/applied
OUT ?= .build
.PHONY: all
all: $(GIT_HOOKS) $(OUT) $(EXEC)

$(GIT_HOOKS):
	@scripts/install-git-hooks
	@echo

CC ?= gcc
CFLAGS = -Wall -std=gnu99 -O2 -I. -msse4.1 -g
LDFLAGS = -lpthread

ifeq ($(strip $(PROFILE)),1)
PROF_FLAGS = -pg
CFLAGS += $(PROF_FLAGS)
LDFLAGS += $(PROF_FLAGS)
endif

OBJS := \
	stopwatch.o \
	matrix_naive.o \
	matrix_sse.o \
	matrix_strassen.o \
	matrix_strassen_sse.o

deps := $(OBJS:%.o=%.o.d)
OBJS := $(addprefix $(OUT)/,$(OBJS))
deps := $(addprefix $(OUT)/,$(deps))

astyle:
	astyle --style=kr --indent=spaces=4 --indent-switches --suffix=none *.[ch]

tests/test-%: $(OBJS) tests/test-%.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(OUT)/%.o: %.c $(OUT)
	$(CC) $(CFLAGS) -c -o $@ -MMD -MF $@.d $<

$(OUT):
	@mkdir -p $@

check: $(EXEC)
	@for test in $^ ; \
	do \
		echo "Execute $$test..." ; $$test && echo "OK!\n" ; \
	done
check-matrix: $(EXEC)
	echo "Execute $<..." ; $< && echo "OK!\n" ;


make plot: check
	gnuplot scripts/runtime.gp

time-check: $(EXEC) check-matrix
	gprof -b $< gmon.out | less

clean:
	$(RM) $(EXEC) $(OBJS) $(deps) record.csv runtime.png gmon.out
	@rm -rf $(OUT)

-include $(deps)
