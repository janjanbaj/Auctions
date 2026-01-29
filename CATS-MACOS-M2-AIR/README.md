## Gemini Aided Guide to Get CATS on MacOS:
The following is a guide with the help of Gemini Pro to compile CATS 2.0 on a MacBook Pro M2.
- As per the README for CATS, remove all instances of -DLINUX flags so that we can compile it for UNIX. 
- Use the MakeFile for CATS as per this repo. Note: I have not used the CPLEX LP Solver for this, rather we rely on the default solver _lp\_solver_ that CATS comes with.

### First Step is Change the CATS/lp_solve/lpglob.h 
This error occurs because modern versions of Flex (used by macOS) define `yyleng` as `unsigned long` (specifically `yy_size_t`), whereas the legacy code in `lpglob.h` expects an `int`.

To fix this, update `kevinlb1/cats/CATS-38cbf35f5c36796522cb76895fd6388f0dd63f5c/lp_solve_4.0/lpglob.h`:

**Change line 17 from:**

```c
extern int yyleng;

```

**To:**

```c
extern unsigned long yyleng;

```

This matches the type expected by the generated `lex.yy.c` on your M2 Mac.

### Second: Change MakeFile Configs for lp_solve:
The linker error `library 'fl' not found` occurs because the Flex library (`-lfl`) is either missing or under a different name on macOS. On macOS, `flex` is included as part of the SDK, but the standalone `-lfl` library is often unnecessary if you define `yywrap` yourself.

To fix this, update `kevinlb1/cats/CATS-38cbf35f5c36796522cb76895fd6388f0dd63f5c/lp_solve_4.0/Makefile`:

1. **Remove the Flex library link:**
Change line 38 from:
```makefile
LEXLIB= -lfl

```


To:
```makefile
LEXLIB=

```


2. **Define `yywrap` in the lexer:**
Since you are no longer linking against the flex library, you must ensure the lexer doesn't call an external `yywrap`. Open `kevinlb1/cats/CATS-38cbf35f5c36796522cb76895fd6388f0dd63f5c/lp_solve_4.0/lex.l` and add the following directive to the first section (at the top):
```c
%option noyywrap

```

Alternatively, if you prefer not to edit the `.l` file, you can change `LEXLIB=` to `-ll` (the standard lex library on macOS/BSD), though `%option noyywrap` is the more modern and portable solution for macOS.

### Third: Update the MakeFile for CATS:
```makefile 
########################################################
# UNCOMMENT/EDIT THESE TO COMPILE WITH CPLEX LIBRARIES
########################################################
# (Comment everything in this section out)
# INCLUDE = ...
# LIB = ...
# LIBDIRS = ...

########################################################
# UNCOMMENT/EDIT THESE TO COMPILE WITH LPSOLVE
########################################################
INCLUDE = -Ilp_solve_4.0
LIB = -lm -llpk
LIBDIRS = -Llp_solve_4.0

RELEASEFLAGS = -O5 -DNDEBUG
DEBUGFLAGS = -Wall -DDEBUG -g
LPMAKE = cd lp_solve_4.0; make;
LPCLEAN = cd lp_solve_4.0; make clean;
```



