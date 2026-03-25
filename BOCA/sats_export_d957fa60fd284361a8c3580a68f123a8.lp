\ENCODING=ISO-8859-1
\Problem name: ilog.cplex

Maximize
 obj1: v__9 + v__8 + v__7 + v__6 + v__5 + v__4 + v__3 + v__2 + v__1 + v__0
Subject To
 c1:    aux_cap_helper__b6,r10,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b6,r10,band_UNPAIRED__Z2 <= 0
 c2:    sv_function_helpervar_b1,r13__Z1 + sv_function_helpervar_b1,r13__Z3
        + sv_function_helpervar_b1,r13__Z2  = 1
 c3:    449.206024367528 aux_sv_b8,r1 - aux_Omega_b8,r1  = 0
 c4:    X__b9,r5,band_UNPAIRED + X__b9,r5,band_HIGH_PAIRED
        + X__b9,r5,band_LOW_PAIRED - GlobalBidder_W_ir__b9,r5 >= 0
 c5:    - 2 aux_cap_helper__b7,r4,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b7,r4,band_HIGH_PAIRED__condX2 <= 0
 c6:    aux_cap_helper__b5,r5,band_UNPAIRED__condX1
        + aux_cap_helper__b5,r5,band_UNPAIRED__condX2 - X__b5,r5,band_UNPAIRED
         = 0
 c7:    - 1.4943850124 aux_cap_helper__b0,r4,band_LOW_PAIRED__Z2
        + 3.735962531 aux_cap_helper__b0,r4,band_LOW_PAIRED__condX1
        + 5.2303475434 aux_cap_helper__b0,r4,band_LOW_PAIRED__condX2
        - aux_cap_b0,r4,band_LOW_PAIRED  = 0
 c8:    - sv_function_helpervar_b7,r10__condX1 <= 0
 c9:    aux_cap_helper__b1,r5,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b1,r5,band_HIGH_PAIRED__Z3 <= 0
 c10:   - aux_cap_helper__b0,r8,band_HIGH_PAIRED__condX1 <= 0
 c11:   - aux_cap_helper__b2,r12,band_UNPAIRED__condX2
        + aux_cap_helper__b2,r12,band_UNPAIRED__Z2 <= 0
 c12:   aux_cap_helper__b2,r10,band_UNPAIRED__condX2
        + aux_cap_helper__b2,r10,band_UNPAIRED__condX1
        - X__b2,r10,band_UNPAIRED  = 0
 c13:   - X__b9,r5,band_HIGH_PAIRED
        + aux_cap_helper__b9,r5,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b9,r5,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b9,r5,band_HIGH_PAIRED__condX2  = 0
 c14:   - X__b6,r1,band_LOW_PAIRED
        + aux_cap_helper__b6,r1,band_LOW_PAIRED__condX2
        + aux_cap_helper__b6,r1,band_LOW_PAIRED__condX1  = 0
 c15:   aux_cap_b9,r7,band_LOW_PAIRED + aux_cap_b9,r7,band_UNPAIRED
        + aux_cap_b9,r7,band_HIGH_PAIRED - aux_c_b9,r7  = 0
 c16:   - aux_cap_helper__b4,r10,band_HIGH_PAIRED__condX1 <= 0
 c17:   - aux_cap_helper__b0,r4,band_LOW_PAIRED__condX1 <= 0
 c18:   - aux_sv_b0,r3 + 4.5976134142 sv_function_helpervar_b0,r3__condX2
        + 1.8850253877 sv_function_helpervar_b0,r3__condX3
        + 109.8706626004 sv_function_helpervar_b0,r3__condX1
        + 34.9363805489559 sv_function_helpervar_b0,r3__Z3
        + 16.8380228584529 sv_function_helpervar_b0,r3__Z2  = 0
 c19:   - aux_cap_helper__b6,r1,band_HIGH_PAIRED__condX1 <= 0
 c20:   - aux_c_b8,r3 + sv_function_helpervar_b8,r3__condX3
        + sv_function_helpervar_b8,r3__condX2
        + sv_function_helpervar_b8,r3__condX1  = 0
 c21:   - aux_cap_helper__b8,r6,band_UNPAIRED__condX1 <= 0
 c22:   aux_cap_helper__b8,r7,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b8,r7,band_HIGH_PAIRED__condX2 <= 0
 c23:   - 2 aux_cap_helper__b2,r12,band_LOW_PAIRED__Z2
        + aux_cap_helper__b2,r12,band_LOW_PAIRED__condX2 <= 0
 c24:   aux_cap_helper__b6,r10,band_UNPAIRED__condX1
        - aux_cap_helper__b6,r10,band_UNPAIRED__Z1 <= 0
 c25:   - sv_function_helpervar_b6,r7__condX3
        + 6.16442568412561 sv_function_helpervar_b6,r7__Z3 <= 0
 c26:   - aux_cap_helper__b7,r4,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b7,r4,band_HIGH_PAIRED__condX1 <= 0
 c27:   518.094789578607 aux_sv_b1,r10 - aux_Omega_b1,r10  = 0
 c28:   aux_cap_helper__b7,r7,band_LOW_PAIRED__Z2
        - aux_cap_helper__b7,r7,band_LOW_PAIRED__condX2 <= 0
 c29:   aux_cap_helper__b1,r13,band_UNPAIRED__Z2
        + aux_cap_helper__b1,r13,band_UNPAIRED__Z1  = 1
 c30:   aux_cap_helper__b8,r6,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b8,r6,band_LOW_PAIRED__Z2 <= 0
 c31:   - aux_cap_helper__b3,r7,band_UNPAIRED__condX1 <= 0
 c32:   7.48967187319432 sv_function_helpervar_b7,r1__Z3
        - sv_function_helpervar_b7,r1__condX3 <= 0
 c33:   - sv_function_helpervar_b4,r3__condX1 <= 0
 c34:   aux_cap_helper__b8,r6,band_UNPAIRED__condX2
        + aux_cap_helper__b8,r6,band_UNPAIRED__condX1 - X__b8,r6,band_UNPAIRED
         = 0
 c35:   - 2 aux_cap_helper__b1,r5,band_UNPAIRED__Z2
        + aux_cap_helper__b1,r5,band_UNPAIRED__condX2 <= 0
 c36:   - sv_function_helpervar_b1,r9__condX3
        + 6.6286062074629 sv_function_helpervar_b1,r9__Z3 <= 0
 c37:   0.159946187454412 sv_function_helpervar_b0,r0__Z2
        - sv_function_helpervar_b0,r0__condX2 <= 0
 c38:   aux_cap_helper__b5,r4,band_UNPAIRED__condX1
        - aux_cap_helper__b5,r4,band_UNPAIRED__Z1 <= 0
 c39:   - aux_cap_helper__b8,r1,band_UNPAIRED__condX2
        + aux_cap_helper__b8,r1,band_UNPAIRED__Z2 <= 0
 c40:   aux_cap_helper__b2,r4,band_LOW_PAIRED__condX1
        + aux_cap_helper__b2,r4,band_LOW_PAIRED__condX2
        - X__b2,r4,band_LOW_PAIRED  = 0
 c41:   - aux_cap_helper__b3,r11,band_LOW_PAIRED__Z1
        + aux_cap_helper__b3,r11,band_LOW_PAIRED__condX1 <= 0
 c42:   1059.4718674214 sv_function_helpervar_b6,r7__condX1
        + 48.0819445271 sv_function_helpervar_b6,r7__condX2
        + 17.2385715004 sv_function_helpervar_b6,r7__condX3
        + 351.89964307056 sv_function_helpervar_b6,r7__Z3
        + 161.767962199713 sv_function_helpervar_b6,r7__Z2 - aux_sv_b6,r7  = 0
 c43:   aux_cap_helper__b2,r4,band_UNPAIRED__condX2
        + aux_cap_helper__b2,r4,band_UNPAIRED__condX1 - X__b2,r4,band_UNPAIRED
         = 0
 c44:   24.5025676163706 sv_function_helpervar_b2,r9__Z2
        + 54.5441408112543 sv_function_helpervar_b2,r9__Z3 - aux_sv_b2,r9
        + 160.8367694004 sv_function_helpervar_b2,r9__condX1
        + 7.6441987844 sv_function_helpervar_b2,r9__condX2
        + 2.5467616526 sv_function_helpervar_b2,r9__condX3  = 0
 c45:   - 2 aux_cap_helper__b3,r11,band_LOW_PAIRED__Z2
        + aux_cap_helper__b3,r11,band_LOW_PAIRED__condX2 <= 0
 c46:   aux_cap_helper__b5,r4,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b5,r4,band_UNPAIRED__Z2 <= 0
 c47:   0.0102040816326531 X__b7,r0,band_UNPAIRED - GlobalBidder_W_ir__b7,r0
        + 0.0102040816326531 X__b7,r0,band_HIGH_PAIRED
        + 0.0102040816326531 X__b7,r0,band_LOW_PAIRED <= 0
 c48:   - aux_cap_helper__b1,r13,band_UNPAIRED__condX1 <= 0
 c49:   - sv_function_helpervar_b4,r8__condX3
        + 6.01793528589529 sv_function_helpervar_b4,r8__Z3 <= 0
 c50:   - aux_cap_helper__b3,r4,band_HIGH_PAIRED__condX1 <= 0
 c51:   aux_cap_helper__b4,r8,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b4,r8,band_HIGH_PAIRED__Z2 <= 0
 c52:   - 15.9946187432626 sv_function_helpervar_b4,r11__Z3
        + sv_function_helpervar_b4,r11__condX3 <= 0
 c53:   - aux_cap_helper__b9,r5,band_LOW_PAIRED__condX1 <= 0
 c54:   - 8.21941784309037 sv_function_helpervar_b9,r8__Z2
        + sv_function_helpervar_b9,r8__condX2 <= 0
 c55:   2 aux_cap_helper__b8,r7,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b8,r7,band_HIGH_PAIRED__condX3 <= 0
 c56:   sv_function_helpervar_b7,r3__condX1
        - 0.159946187436532 sv_function_helpervar_b7,r3__Z1 <= 0
 c57:   aux_cap_b9,r11,band_HIGH_PAIRED + aux_cap_b9,r11,band_LOW_PAIRED
        + aux_cap_b9,r11,band_UNPAIRED - aux_c_b9,r11  = 0
 c58:   aux_cap_helper__b2,r11,band_UNPAIRED__Z1
        + aux_cap_helper__b2,r11,band_UNPAIRED__Z2  = 1
 c59:   aux_cap_helper__b6,r12,band_LOW_PAIRED__Z2
        - aux_cap_helper__b6,r12,band_LOW_PAIRED__condX2 <= 0
 c60:   5.2303475434 aux_cap_helper__b7,r3,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b7,r3,band_LOW_PAIRED__condX1
        - aux_cap_b7,r3,band_LOW_PAIRED
        - 1.4943850124 aux_cap_helper__b7,r3,band_LOW_PAIRED__Z2  = 0
 c61:   aux_cap_helper__b3,r1,band_LOW_PAIRED__condX1
        - aux_cap_helper__b3,r1,band_LOW_PAIRED__Z1 <= 0
 c62:   - 3 aux_cap_helper__b3,r12,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b3,r12,band_HIGH_PAIRED__condX3 <= 0
 c63:   sv_function_helpervar_b5,r9__condX1
        - 0.15994618751673 sv_function_helpervar_b5,r9__Z1 <= 0
 c64:   aux_cap_b8,r1,band_HIGH_PAIRED + aux_cap_b8,r1,band_LOW_PAIRED
        - aux_c_b8,r1 + aux_cap_b8,r1,band_UNPAIRED  = 0
 c65:   aux_cap_helper__b7,r12,band_UNPAIRED__Z2
        + aux_cap_helper__b7,r12,band_UNPAIRED__Z1  = 1
 c66:   - X__b0,r13,band_UNPAIRED
        + aux_cap_helper__b0,r13,band_UNPAIRED__condX2
        + aux_cap_helper__b0,r13,band_UNPAIRED__condX1  = 0
 c67:   - aux_cap_helper__b0,r10,band_LOW_PAIRED__condX1 <= 0
 c68:   aux_cap_helper__b3,r11,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b3,r11,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b3,r11,band_HIGH_PAIRED__condX1
        - X__b3,r11,band_HIGH_PAIRED  = 0
 c69:   2 aux_cap_helper__b8,r8,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b8,r8,band_HIGH_PAIRED__condX3 <= 0
 c70:   - aux_cap_helper__b6,r0,band_HIGH_PAIRED__condX1 <= 0
 c71:   - aux_cap_helper__b2,r2,band_LOW_PAIRED__condX2
        + aux_cap_helper__b2,r2,band_LOW_PAIRED__Z2 <= 0
 c72:   1.1250244065 aux_cap_helper__b5,r1,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b5,r1,band_UNPAIRED__condX1
        - aux_cap_b5,r1,band_UNPAIRED
        - 0.3214355447 aux_cap_helper__b5,r1,band_UNPAIRED__Z2  = 0
 c73:   aux_cap_helper__b1,r4,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b1,r4,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b1,r4,band_HIGH_PAIRED__condX3
        - X__b1,r4,band_HIGH_PAIRED  = 0
 c74:   aux_cap_helper__b3,r1,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b3,r1,band_LOW_PAIRED__Z2 <= 0
 c75:   X__b7,r8,band_UNPAIRED - GlobalBidder_W_ir__b7,r8
        + X__b7,r8,band_HIGH_PAIRED + X__b7,r8,band_LOW_PAIRED >= 0
 c76:   - 15.9946187432626 sv_function_helpervar_b9,r2__Z3
        + sv_function_helpervar_b9,r2__condX3 <= 0
 c77:   aux_cap_helper__b6,r2,band_UNPAIRED__Z2
        + aux_cap_helper__b6,r2,band_UNPAIRED__Z1  = 1
 c78:   aux_cap_helper__b3,r6,band_UNPAIRED__Z2
        + aux_cap_helper__b3,r6,band_UNPAIRED__Z1  = 1
 c79:   aux_cap_helper__b3,r11,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b3,r11,band_HIGH_PAIRED__Z3 <= 0
 c80:   - 3 aux_cap_helper__b8,r10,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b8,r10,band_HIGH_PAIRED__condX3 <= 0
 c81:   - aux_cap_helper__b4,r7,band_UNPAIRED__condX1 <= 0
 c82:   X__b1,r3,band_LOW_PAIRED + X__b7,r3,band_LOW_PAIRED
        + X__b8,r3,band_LOW_PAIRED + X__b9,r3,band_LOW_PAIRED
        + X__b0,r3,band_LOW_PAIRED + X__b6,r3,band_LOW_PAIRED
        + X__b5,r3,band_LOW_PAIRED + X__b4,r3,band_LOW_PAIRED
        + X__b3,r3,band_LOW_PAIRED + X__b2,r3,band_LOW_PAIRED <= 2
 c83:   aux_cap_helper__b4,r9,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b4,r9,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b4,r9,band_HIGH_PAIRED__Z3  = 1
 c84:   8.28541025644105 sv_function_helpervar_b8,r3__Z3
        - sv_function_helpervar_b8,r3__condX3 <= 0
 c85:   - 6.124193939318 sv_function_helpervar_b2,r12__Z2
        + sv_function_helpervar_b2,r12__condX2 <= 0
 c86:   - aux_cap_helper__b5,r9,band_LOW_PAIRED__condX1 <= 0
 c87:   - sv_function_helpervar_b1,r9__condX1 <= 0
 c88:   5.2303475434 aux_cap_helper__b7,r13,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b7,r13,band_LOW_PAIRED__condX1
        - aux_cap_b7,r13,band_LOW_PAIRED
        - 1.4943850124 aux_cap_helper__b7,r13,band_LOW_PAIRED__Z2  = 0
 c89:   sv_function_helpervar_b7,r6__condX3
        - 15.9946187432626 sv_function_helpervar_b7,r6__Z3 <= 0
 c90:   X__b8,r12,band_UNPAIRED + X__b8,r12,band_HIGH_PAIRED
        - GlobalBidder_W_ir__b8,r12 + X__b8,r12,band_LOW_PAIRED >= 0
 c91:   - 2 aux_cap_helper__b7,r5,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b7,r5,band_HIGH_PAIRED__condX2 <= 0
 c92:   - aux_cap_helper__b6,r4,band_UNPAIRED__Z1
        + aux_cap_helper__b6,r4,band_UNPAIRED__condX1 <= 0
 c93:   - aux_cap_helper__b7,r1,band_UNPAIRED__condX2
        + aux_cap_helper__b7,r1,band_UNPAIRED__Z2 <= 0
 c94:   aux_cap_helper__b3,r4,band_LOW_PAIRED__condX1
        - aux_cap_helper__b3,r4,band_LOW_PAIRED__Z1 <= 0
 c95:   - aux_Omega_b2,r8 + 254.1510853109 aux_sv_b2,r8  = 0
 c96:   aux_cap_helper__b3,r13,band_UNPAIRED__condX2
        + aux_cap_helper__b3,r13,band_UNPAIRED__condX1
        - X__b3,r13,band_UNPAIRED  = 0
 c97:   sv_function_helpervar_b8,r12__condX2
        + sv_function_helpervar_b8,r12__condX3
        + sv_function_helpervar_b8,r12__condX1 - aux_c_b8,r12  = 0
 c98:   - sv_function_helpervar_b3,r5__condX1 <= 0
 c99:   - aux_cap_helper__b4,r3,band_LOW_PAIRED__condX1 <= 0
 c100:  sv_function_helpervar_b0,r11__Z1 + sv_function_helpervar_b0,r11__Z2
        + sv_function_helpervar_b0,r11__Z3  = 1
 c101:  - 2 aux_cap_helper__b4,r7,band_LOW_PAIRED__Z2
        + aux_cap_helper__b4,r7,band_LOW_PAIRED__condX2 <= 0
 c102:  0.159946187471719 sv_function_helpervar_b2,r13__Z2
        - sv_function_helpervar_b2,r13__condX2 <= 0
 c103:  - 6.18724675079566 sv_function_helpervar_b3,r11__Z2
        + sv_function_helpervar_b3,r11__condX2 <= 0
 c104:  aux_cap_helper__b5,r13,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b5,r13,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b5,r13,band_HIGH_PAIRED__condX1
        - X__b5,r13,band_HIGH_PAIRED  = 0
 c105:  aux_cap_helper__b8,r8,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b8,r8,band_HIGH_PAIRED__condX2 <= 0
 c106:  - aux_cap_helper__b0,r9,band_HIGH_PAIRED__condX1 <= 0
 c107:  aux_cap_helper__b1,r6,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b1,r6,band_HIGH_PAIRED__Z3 <= 0
 c108:  - 0.3214355447 aux_cap_helper__b9,r11,band_UNPAIRED__Z2
        + 0.8035888618 aux_cap_helper__b9,r11,band_UNPAIRED__condX1
        + 1.1250244065 aux_cap_helper__b9,r11,band_UNPAIRED__condX2
        - aux_cap_b9,r11,band_UNPAIRED  = 0
 c109:  - aux_cap_helper__b2,r2,band_UNPAIRED__condX2
        + aux_cap_helper__b2,r2,band_UNPAIRED__Z2 <= 0
 c110:  sv_function_helpervar_b8,r2__condX2
        - 8.31036492628082 sv_function_helpervar_b8,r2__Z2 <= 0
 c111:  - sv_function_helpervar_b0,r11__condX1 <= 0
 c112:  aux_cap_helper__b1,r13,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b1,r13,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b1,r13,band_HIGH_PAIRED__Z3  = 1
 c113:  - aux_cap_helper__b3,r8,band_LOW_PAIRED__condX2
        + aux_cap_helper__b3,r8,band_LOW_PAIRED__Z2 <= 0
 c114:  - 3 aux_cap_helper__b4,r2,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b4,r2,band_HIGH_PAIRED__condX3 <= 0
 c115:  - aux_cap_helper__b5,r13,band_LOW_PAIRED__condX2
        + aux_cap_helper__b5,r13,band_LOW_PAIRED__Z2 <= 0
 c116:  sv_function_helpervar_b6,r9__Z3 + sv_function_helpervar_b6,r9__Z2
        + sv_function_helpervar_b6,r9__Z1  = 1
 c117:  - 2 aux_cap_helper__b7,r0,band_LOW_PAIRED__Z2
        + aux_cap_helper__b7,r0,band_LOW_PAIRED__condX2 <= 0
 c118:  - aux_sv_b2,r1 + 160.8367694197 sv_function_helpervar_b2,r1__condX1
        + 2.5489144814 sv_function_helpervar_b2,r1__condX3
        + 7.6328410933 sv_function_helpervar_b2,r1__condX2
        + 24.5043842359091 sv_function_helpervar_b2,r1__Z2
        + 54.5097071353929 sv_function_helpervar_b2,r1__Z3  = 0
 c119:  aux_cap_helper__b4,r8,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b4,r8,band_HIGH_PAIRED__Z1 <= 0
 c120:  - aux_cap_helper__b7,r6,band_UNPAIRED__condX1 <= 0
 c121:  aux_cap_b6,r4,band_LOW_PAIRED - aux_c_b6,r4
        + aux_cap_b6,r4,band_HIGH_PAIRED + aux_cap_b6,r4,band_UNPAIRED  = 0
 c122:  - aux_cap_helper__b4,r10,band_LOW_PAIRED__Z1
        + aux_cap_helper__b4,r10,band_LOW_PAIRED__condX1 <= 0
 c123:  - aux_cap_helper__b7,r5,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b7,r5,band_HIGH_PAIRED__condX1 <= 0
 c124:  - aux_Omega_b7,r3 + 694.938687182136 aux_sv_b7,r3  = 0
 c125:  - aux_cap_helper__b0,r5,band_UNPAIRED__Z1
        + aux_cap_helper__b0,r5,band_UNPAIRED__condX1 <= 0
 c126:  aux_cap_helper__b2,r3,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b2,r3,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b2,r3,band_HIGH_PAIRED__Z2  = 1
 c127:  - sv_function_helpervar_b7,r8__condX1 <= 0
 c128:  sv_function_helpervar_b3,r0__Z3 + sv_function_helpervar_b3,r0__Z2
        + sv_function_helpervar_b3,r0__Z1  = 1
 c129:  - 2 aux_cap_helper__b4,r10,band_LOW_PAIRED__Z2
        + aux_cap_helper__b4,r10,band_LOW_PAIRED__condX2 <= 0
 c130:  - 2 aux_cap_helper__b0,r5,band_UNPAIRED__Z2
        + aux_cap_helper__b0,r5,band_UNPAIRED__condX2 <= 0
 c131:  aux_cap_helper__b8,r6,band_LOW_PAIRED__condX1
        - aux_cap_helper__b8,r6,band_LOW_PAIRED__Z1 <= 0
 c132:  - X__b8,r13,band_LOW_PAIRED
        + aux_cap_helper__b8,r13,band_LOW_PAIRED__condX2
        + aux_cap_helper__b8,r13,band_LOW_PAIRED__condX1  = 0
 c133:  - sv_function_helpervar_b2,r7__condX1 <= 0
 c134:  - aux_cap_helper__b7,r0,band_LOW_PAIRED__Z1
        + aux_cap_helper__b7,r0,band_LOW_PAIRED__condX1 <= 0
 c135:  - 2 aux_cap_helper__b6,r4,band_UNPAIRED__Z2
        + aux_cap_helper__b6,r4,band_UNPAIRED__condX2 <= 0
 c136:  sv_function_helpervar_b8,r4__condX3
        - 15.9946187432626 sv_function_helpervar_b8,r4__Z3 <= 0
 c137:  aux_cap_helper__b1,r9,band_UNPAIRED__Z2
        + aux_cap_helper__b1,r9,band_UNPAIRED__Z1  = 1
 c138:  - aux_cap_helper__b3,r3,band_HIGH_PAIRED__condX1 <= 0
 c139:  X__b8,r10,band_HIGH_PAIRED + X__b5,r10,band_HIGH_PAIRED
        + X__b7,r10,band_HIGH_PAIRED + X__b1,r10,band_HIGH_PAIRED
        + X__b9,r10,band_HIGH_PAIRED + X__b4,r10,band_HIGH_PAIRED
        + X__b6,r10,band_HIGH_PAIRED + X__b3,r10,band_HIGH_PAIRED
        + X__b2,r10,band_HIGH_PAIRED + X__b0,r10,band_HIGH_PAIRED <= 3
 c140:  1.1250244065 aux_cap_helper__b6,r10,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b6,r10,band_UNPAIRED__condX1
        - 0.3214355447 aux_cap_helper__b6,r10,band_UNPAIRED__Z2
        - aux_cap_b6,r10,band_UNPAIRED  = 0
 c141:  430.195243506472 aux_sv_b8,r9 - aux_Omega_b8,r9  = 0
 c142:  aux_cap_helper__b8,r3,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b8,r3,band_LOW_PAIRED__Z2 <= 0
 c143:  0.159946187488557 sv_function_helpervar_b9,r2__Z2
        - sv_function_helpervar_b9,r2__condX2 <= 0
 c144:  X__b9,r10,band_LOW_PAIRED + X__b9,r10,band_UNPAIRED
        - GlobalBidder_W_ir__b9,r10 + X__b9,r10,band_HIGH_PAIRED >= 0
 c145:  aux_cap_helper__b0,r2,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b0,r2,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b0,r2,band_HIGH_PAIRED__Z1  = 1
 c146:  - 6.07097130497179 sv_function_helpervar_b1,r13__Z2
        + sv_function_helpervar_b1,r13__condX2 <= 0
 c147:  sv_function_helpervar_b1,r6__condX3
        - 15.9946187432626 sv_function_helpervar_b1,r6__Z3 <= 0
 c148:  - 0.3214355447 aux_cap_helper__b8,r7,band_UNPAIRED__Z2
        - aux_cap_b8,r7,band_UNPAIRED
        + 0.8035888618 aux_cap_helper__b8,r7,band_UNPAIRED__condX1
        + 1.1250244065 aux_cap_helper__b8,r7,band_UNPAIRED__condX2  = 0
 c149:  178.383670139669 sv_function_helpervar_b5,r6__Z2
        + 392.678401570576 sv_function_helpervar_b5,r6__Z3 - aux_sv_b5,r6
        + 1169.5974415305 sv_function_helpervar_b5,r6__condX1
        + 54.3244058643 sv_function_helpervar_b5,r6__condX2
        + 18.7677666091 sv_function_helpervar_b5,r6__condX3  = 0
 c150:  - 1.4943850124 aux_cap_helper__b0,r10,band_LOW_PAIRED__Z2
        + 5.2303475434 aux_cap_helper__b0,r10,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b0,r10,band_LOW_PAIRED__condX1
        - aux_cap_b0,r10,band_LOW_PAIRED  = 0
 c151:  233.120606977903 aux_sv_b0,r2 - aux_Omega_b0,r2  = 0
 c152:  - aux_Omega_b6,r10 + 418.047330869981 aux_sv_b6,r10  = 0
 c153:  aux_cap_helper__b6,r5,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b6,r5,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b6,r5,band_HIGH_PAIRED__Z3  = 1
 c154:  - aux_cap_b9,r10,band_HIGH_PAIRED
        - 0.6181448971 aux_cap_helper__b9,r10,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b9,r10,band_HIGH_PAIRED__Z3
        + 1.5453622427 aux_cap_helper__b9,r10,band_HIGH_PAIRED__condX1
        + 2.1635071398 aux_cap_helper__b9,r10,band_HIGH_PAIRED__condX2
        + 1.3908260184 aux_cap_helper__b9,r10,band_HIGH_PAIRED__condX3  = 0
 c155:  - 0.159946187488921 sv_function_helpervar_b2,r11__Z1
        + sv_function_helpervar_b2,r11__condX1 <= 0
 c156:  aux_cap_b0,r2,band_HIGH_PAIRED + aux_cap_b0,r2,band_UNPAIRED
        - aux_c_b0,r2 + aux_cap_b0,r2,band_LOW_PAIRED  = 0
 c157:  sv_function_helpervar_b7,r6__condX1
        - 0.159946187459757 sv_function_helpervar_b7,r6__Z1 <= 0
 c158:  aux_cap_helper__b7,r0,band_UNPAIRED__Z1
        + aux_cap_helper__b7,r0,band_UNPAIRED__Z2  = 1
 c159:  sv_function_helpervar_b0,r7__Z1 + sv_function_helpervar_b0,r7__Z2
        + sv_function_helpervar_b0,r7__Z3  = 1
 c160:  2 aux_cap_helper__b8,r9,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b8,r9,band_HIGH_PAIRED__condX3 <= 0
 c161:  aux_cap_helper__b8,r10,band_UNPAIRED__condX2
        + aux_cap_helper__b8,r10,band_UNPAIRED__condX1
        - X__b8,r10,band_UNPAIRED  = 0
 c162:  - aux_sv_b1,r11 + 125.5635947085 sv_function_helpervar_b1,r11__condX1
        + 5.9144733796 sv_function_helpervar_b1,r11__condX2
        + 1.9984476514 sv_function_helpervar_b1,r11__condX3
        + 42.4186223260833 sv_function_helpervar_b1,r11__Z3
        + 19.1374207908736 sv_function_helpervar_b1,r11__Z2  = 0
 c163:  - aux_cap_helper__b4,r12,band_UNPAIRED__Z1
        + aux_cap_helper__b4,r12,band_UNPAIRED__condX1 <= 0
 c164:  0.8035888618 aux_cap_helper__b3,r6,band_UNPAIRED__condX1
        + 1.1250244065 aux_cap_helper__b3,r6,band_UNPAIRED__condX2
        - 0.3214355447 aux_cap_helper__b3,r6,band_UNPAIRED__Z2
        - aux_cap_b3,r6,band_UNPAIRED  = 0
 c165:  0.8035888618 aux_cap_helper__b6,r8,band_UNPAIRED__condX1
        + 1.1250244065 aux_cap_helper__b6,r8,band_UNPAIRED__condX2
        - aux_cap_b6,r8,band_UNPAIRED
        - 0.3214355447 aux_cap_helper__b6,r8,band_UNPAIRED__Z2  = 0
 c166:  2 aux_cap_helper__b5,r9,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b5,r9,band_HIGH_PAIRED__condX3 <= 0
 c167:  aux_cap_helper__b3,r11,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b3,r11,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b3,r11,band_HIGH_PAIRED__Z3  = 1
 c168:  aux_cap_helper__b7,r2,band_UNPAIRED__condX1
        + aux_cap_helper__b7,r2,band_UNPAIRED__condX2 - X__b7,r2,band_UNPAIRED
         = 0
 c169:  aux_cap_helper__b8,r3,band_LOW_PAIRED__condX1
        - aux_cap_helper__b8,r3,band_LOW_PAIRED__Z1 <= 0
 c170:  - sv_function_helpervar_b9,r4__condX1 <= 0
 c171:  aux_cap_helper__b4,r5,band_UNPAIRED__Z1
        + aux_cap_helper__b4,r5,band_UNPAIRED__Z2  = 1
 c172:  aux_cap_helper__b9,r9,band_LOW_PAIRED__condX1
        - aux_cap_helper__b9,r9,band_LOW_PAIRED__Z1 <= 0
 c173:  - X__b7,r3,band_LOW_PAIRED
        + aux_cap_helper__b7,r3,band_LOW_PAIRED__condX2
        + aux_cap_helper__b7,r3,band_LOW_PAIRED__condX1  = 0
 c174:  - aux_c_b0,r11 + sv_function_helpervar_b0,r11__condX2
        + sv_function_helpervar_b0,r11__condX1
        + sv_function_helpervar_b0,r11__condX3  = 0
 c175:  - aux_cap_helper__b6,r2,band_HIGH_PAIRED__condX1 <= 0
 c176:  - 15.9946187432626 sv_function_helpervar_b3,r2__Z3
        + sv_function_helpervar_b3,r2__condX3 <= 0
 c177:  5.79966606576978 sv_function_helpervar_b4,r6__Z3
        - sv_function_helpervar_b4,r6__condX3 <= 0
 c178:  - aux_Omega_b3,r10 + 457.692860835036 aux_sv_b3,r10  = 0
 c179:  aux_cap_helper__b2,r9,band_LOW_PAIRED__Z2
        + aux_cap_helper__b2,r9,band_LOW_PAIRED__Z1  = 1
 c180:  aux_cap_b8,r13,band_UNPAIRED + aux_cap_b8,r13,band_LOW_PAIRED
        - aux_c_b8,r13 + aux_cap_b8,r13,band_HIGH_PAIRED  = 0
 c181:  - X__b7,r4,band_HIGH_PAIRED
        + aux_cap_helper__b7,r4,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b7,r4,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b7,r4,band_HIGH_PAIRED__condX3  = 0
 c182:  aux_cap_helper__b9,r9,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b9,r9,band_LOW_PAIRED__Z2 <= 0
 c183:  - aux_cap_helper__b0,r1,band_LOW_PAIRED__condX1 <= 0
 c184:  sv_function_helpervar_b9,r6__condX1
        - 0.159946187446589 sv_function_helpervar_b9,r6__Z1 <= 0
 c185:  - 15.9946187432626 sv_function_helpervar_b6,r8__Z3
        + sv_function_helpervar_b6,r8__condX3 <= 0
 c186:  - aux_cap_helper__b4,r11,band_HIGH_PAIRED__condX1 <= 0
 c187:  - 1.4943850124 aux_cap_helper__b8,r13,band_LOW_PAIRED__Z2
        - aux_cap_b8,r13,band_LOW_PAIRED
        + 5.2303475434 aux_cap_helper__b8,r13,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b8,r13,band_LOW_PAIRED__condX1  = 0
 c188:  - aux_cap_helper__b9,r8,band_LOW_PAIRED__condX1 <= 0
 c189:  aux_cap_helper__b6,r13,band_LOW_PAIRED__Z2
        + aux_cap_helper__b6,r13,band_LOW_PAIRED__Z1  = 1
 c190:  - aux_Omega_b7,r1 + 539.810871990007 aux_sv_b7,r1  = 0
 c191:  14 GlobalBidder_hat{W}__i8,k2 + GlobalBidder_W_i_8 <= 26
 c192:  aux_Omega_b9,r1 + aux_Omega_b9,r0 + aux_Omega_b9,r3 + aux_Omega_b9,r2
        + aux_Omega_b9,r5 + aux_Omega_b9,r4 + aux_Omega_b9,r7 + aux_Omega_b9,r6
        + aux_Omega_b9,r9 + aux_Omega_b9,r8 - GlobalBidder_PSI__i9,k2
        + aux_Omega_b9,r13 + aux_Omega_b9,r10 + aux_Omega_b9,r11
        + aux_Omega_b9,r12 + 18497526.8940961 GlobalBidder_hat{W}__i9,k2
        <= 18497526.8940961
 c193:  aux_cap_helper__b6,r4,band_LOW_PAIRED__Z2
        - aux_cap_helper__b6,r4,band_LOW_PAIRED__condX2 <= 0
 c194:  - sv_function_helpervar_b9,r3__condX2
        + 0.159946187444714 sv_function_helpervar_b9,r3__Z2 <= 0
 c195:  - X__b1,r9,band_UNPAIRED + aux_cap_helper__b1,r9,band_UNPAIRED__condX2
        + aux_cap_helper__b1,r9,band_UNPAIRED__condX1  = 0
 c196:  - X__b4,r0,band_HIGH_PAIRED
        + aux_cap_helper__b4,r0,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b4,r0,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b4,r0,band_HIGH_PAIRED__condX1  = 0
 c197:  aux_cap_helper__b4,r12,band_LOW_PAIRED__Z2
        + aux_cap_helper__b4,r12,band_LOW_PAIRED__Z1  = 1
 c198:  - aux_Omega_b4,r0 + 114.275219873322 aux_sv_b4,r0  = 0
 c199:  aux_cap_helper__b7,r2,band_LOW_PAIRED__Z1
        + aux_cap_helper__b7,r2,band_LOW_PAIRED__Z2  = 1
 c200:  - 3 aux_cap_helper__b8,r11,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b8,r11,band_HIGH_PAIRED__condX3 <= 0
 c201:  sv_function_helpervar_b2,r4__Z1 + sv_function_helpervar_b2,r4__Z2
        + sv_function_helpervar_b2,r4__Z3  = 1
 c202:  - aux_cap_b2,r5,band_LOW_PAIRED
        - 1.4943850124 aux_cap_helper__b2,r5,band_LOW_PAIRED__Z2
        + 3.735962531 aux_cap_helper__b2,r5,band_LOW_PAIRED__condX1
        + 5.2303475434 aux_cap_helper__b2,r5,band_LOW_PAIRED__condX2  = 0
 c203:  - X__b3,r6,band_LOW_PAIRED
        + aux_cap_helper__b3,r6,band_LOW_PAIRED__condX1
        + aux_cap_helper__b3,r6,band_LOW_PAIRED__condX2  = 0
 c204:  - 2 aux_cap_helper__b4,r12,band_UNPAIRED__Z2
        + aux_cap_helper__b4,r12,band_UNPAIRED__condX2 <= 0
 c205:  - aux_cap_helper__b1,r7,band_LOW_PAIRED__condX1 <= 0
 c206:  sv_function_helpervar_b2,r10__Z3 + sv_function_helpervar_b2,r10__Z2
        + sv_function_helpervar_b2,r10__Z1  = 1
 c207:  - aux_cap_b5,r2,band_LOW_PAIRED
        + 5.2303475434 aux_cap_helper__b5,r2,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b5,r2,band_LOW_PAIRED__condX1
        - 1.4943850124 aux_cap_helper__b5,r2,band_LOW_PAIRED__Z2  = 0
 c208:  aux_cap_b3,r1,band_LOW_PAIRED + aux_cap_b3,r1,band_HIGH_PAIRED
        - aux_c_b3,r1 + aux_cap_b3,r1,band_UNPAIRED  = 0
 c209:  sv_function_helpervar_b6,r13__condX3
        - 15.9946187432626 sv_function_helpervar_b6,r13__Z3 <= 0
 c210:  6.10406421692187 sv_function_helpervar_b5,r11__Z3
        - sv_function_helpervar_b5,r11__condX3 <= 0
 c211:  - X__b5,r10,band_UNPAIRED
        + aux_cap_helper__b5,r10,band_UNPAIRED__condX2
        + aux_cap_helper__b5,r10,band_UNPAIRED__condX1  = 0
 c212:  - aux_cap_helper__b3,r2,band_HIGH_PAIRED__condX1 <= 0
 c213:  aux_cap_helper__b6,r9,band_UNPAIRED__condX1
        + aux_cap_helper__b6,r9,band_UNPAIRED__condX2 - X__b6,r9,band_UNPAIRED
         = 0
 c214:  aux_cap_helper__b1,r1,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b1,r1,band_HIGH_PAIRED__condX2 <= 0
 c215:  0.159946187528613 sv_function_helpervar_b1,r5__Z2
        - sv_function_helpervar_b1,r5__condX2 <= 0
 c216:  - aux_cap_helper__b1,r13,band_LOW_PAIRED__Z1
        + aux_cap_helper__b1,r13,band_LOW_PAIRED__condX1 <= 0
 c217:  - X__b0,r13,band_LOW_PAIRED
        + aux_cap_helper__b0,r13,band_LOW_PAIRED__condX1
        + aux_cap_helper__b0,r13,band_LOW_PAIRED__condX2  = 0
 c218:  - aux_c_b2,r13 + sv_function_helpervar_b2,r13__condX1
        + sv_function_helpervar_b2,r13__condX2
        + sv_function_helpervar_b2,r13__condX3  = 0
 c219:  - 0.3214355447 aux_cap_helper__b0,r8,band_UNPAIRED__Z2
        + 1.1250244065 aux_cap_helper__b0,r8,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b0,r8,band_UNPAIRED__condX1
        - aux_cap_b0,r8,band_UNPAIRED  = 0
 c220:  390.335082339874 sv_function_helpervar_b5,r11__Z3
        + 178.496531968884 sv_function_helpervar_b5,r11__Z2 - aux_sv_b5,r11
        + 1169.597442207 sv_function_helpervar_b5,r11__condX1
        + 18.9142733354 sv_function_helpervar_b5,r11__condX3
        + 53.6187821418 sv_function_helpervar_b5,r11__condX2  = 0
 c221:  aux_cap_helper__b8,r4,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b8,r4,band_UNPAIRED__Z2 <= 0
 c222:  aux_cap_helper__b0,r0,band_UNPAIRED__Z2
        + aux_cap_helper__b0,r0,band_UNPAIRED__Z1  = 1
 c223:  2 aux_cap_helper__b1,r1,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b1,r1,band_HIGH_PAIRED__condX3 <= 0
 c224:  - 2 aux_cap_helper__b1,r13,band_LOW_PAIRED__Z2
        + aux_cap_helper__b1,r13,band_LOW_PAIRED__condX2 <= 0
 c225:  aux_cap_helper__b7,r2,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b7,r2,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b7,r2,band_HIGH_PAIRED__Z2  = 1
 c226:  sv_function_helpervar_b8,r1__Z3 + sv_function_helpervar_b8,r1__Z2
        + sv_function_helpervar_b8,r1__Z1  = 1
 c227:  - aux_cap_helper__b4,r7,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b4,r7,band_HIGH_PAIRED__condX1 <= 0
 c228:  - aux_Omega_b5,r6 + 44.1504639496101 aux_sv_b5,r6  = 0
 c229:  - aux_cap_helper__b7,r12,band_UNPAIRED__condX2
        + aux_cap_helper__b7,r12,band_UNPAIRED__Z2 <= 0
 c230:  0.159946187449246 sv_function_helpervar_b3,r6__Z2
        - sv_function_helpervar_b3,r6__condX2 <= 0
 c231:  0.159946187436407 sv_function_helpervar_b8,r13__Z2
        - sv_function_helpervar_b8,r13__condX2 <= 0
 c232:  - aux_cap_helper__b1,r5,band_UNPAIRED__Z1
        + aux_cap_helper__b1,r5,band_UNPAIRED__condX1 <= 0
 c233:  aux_cap_b4,r10,band_HIGH_PAIRED + aux_cap_b4,r10,band_LOW_PAIRED
        - aux_c_b4,r10 + aux_cap_b4,r10,band_UNPAIRED  = 0
 c234:  3.735962531 aux_cap_helper__b3,r8,band_LOW_PAIRED__condX1
        + 5.2303475434 aux_cap_helper__b3,r8,band_LOW_PAIRED__condX2
        - aux_cap_b3,r8,band_LOW_PAIRED
        - 1.4943850124 aux_cap_helper__b3,r8,band_LOW_PAIRED__Z2  = 0
 c235:  - X__b4,r12,band_HIGH_PAIRED
        + aux_cap_helper__b4,r12,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b4,r12,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b4,r12,band_HIGH_PAIRED__condX1  = 0
 c236:  - aux_cap_helper__b3,r5,band_HIGH_PAIRED__condX1 <= 0
 c237:  - X__b6,r9,band_HIGH_PAIRED
        + aux_cap_helper__b6,r9,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b6,r9,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b6,r9,band_HIGH_PAIRED__condX1  = 0
 c238:  sv_function_helpervar_b6,r13__condX1
        - 0.159946187451606 sv_function_helpervar_b6,r13__Z1 <= 0
 c239:  - sv_function_helpervar_b8,r8__condX3
        + 6.66714254024684 sv_function_helpervar_b8,r8__Z3 <= 0
 c240:  - aux_cap_helper__b3,r7,band_LOW_PAIRED__Z1
        + aux_cap_helper__b3,r7,band_LOW_PAIRED__condX1 <= 0
 c241:  - X__b0,r9,band_HIGH_PAIRED
        + aux_cap_helper__b0,r9,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b0,r9,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b0,r9,band_HIGH_PAIRED__condX1  = 0
 c242:  aux_cap_helper__b9,r10,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b9,r10,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b9,r10,band_HIGH_PAIRED__Z1  = 1
 c243:  434.628336009163 sv_function_helpervar_b4,r0__Z3
        + 196.557108786124 sv_function_helpervar_b4,r0__Z2
        + 60.4320211016 sv_function_helpervar_b4,r0__condX2
        + 1289.3272634202 sv_function_helpervar_b4,r0__condX1
        + 20.5794514197 sv_function_helpervar_b4,r0__condX3 - aux_sv_b4,r0  = 0
 c244:  6.34405022357509 sv_function_helpervar_b0,r13__Z3
        - sv_function_helpervar_b0,r13__condX3 <= 0
 c245:  - 3 aux_cap_helper__b1,r4,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b1,r4,band_HIGH_PAIRED__condX3 <= 0
 c246:  aux_cap_helper__b8,r6,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b8,r6,band_HIGH_PAIRED__condX2 <= 0
 c247:  sv_function_helpervar_b1,r12__condX2
        + sv_function_helpervar_b1,r12__condX1
        + sv_function_helpervar_b1,r12__condX3 - aux_c_b1,r12  = 0
 c248:  - aux_cap_helper__b5,r12,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b5,r12,band_HIGH_PAIRED__Z3 <= 0
 c249:  6.19171029678876 sv_function_helpervar_b2,r11__Z3
        - sv_function_helpervar_b2,r11__condX3 <= 0
 c250:  - 3 aux_cap_helper__b1,r7,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b1,r7,band_HIGH_PAIRED__condX3 <= 0
 c251:  aux_cap_helper__b8,r9,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b8,r9,band_HIGH_PAIRED__condX2 <= 0
 c252:  aux_cap_helper__b4,r4,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b4,r4,band_LOW_PAIRED__Z2 <= 0
 c253:  aux_cap_helper__b6,r7,band_LOW_PAIRED__Z1
        + aux_cap_helper__b6,r7,band_LOW_PAIRED__Z2  = 1
 c254:  - aux_cap_helper__b6,r7,band_UNPAIRED__condX1 <= 0
 c255:  2 aux_cap_helper__b8,r6,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b8,r6,band_HIGH_PAIRED__condX3 <= 0
 c256:  - aux_cap_helper__b3,r2,band_UNPAIRED__condX2
        + aux_cap_helper__b3,r2,band_UNPAIRED__Z2 <= 0
 c257:  aux_cap_helper__b4,r1,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b4,r1,band_HIGH_PAIRED__Z3 <= 0
 c258:  - aux_cap_b9,r13,band_LOW_PAIRED
        - 1.4943850124 aux_cap_helper__b9,r13,band_LOW_PAIRED__Z2
        + 5.2303475434 aux_cap_helper__b9,r13,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b9,r13,band_LOW_PAIRED__condX1  = 0
 c259:  aux_cap_helper__b4,r13,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b4,r13,band_LOW_PAIRED__Z2 <= 0
 c260:  aux_cap_helper__b5,r9,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b5,r9,band_HIGH_PAIRED__condX2 <= 0
 c261:  - aux_cap_helper__b2,r12,band_LOW_PAIRED__Z1
        + aux_cap_helper__b2,r12,band_LOW_PAIRED__condX1 <= 0
 c262:  aux_cap_helper__b2,r5,band_LOW_PAIRED__Z2
        - aux_cap_helper__b2,r5,band_LOW_PAIRED__condX2 <= 0
 c263:  - aux_cap_b4,r11,band_HIGH_PAIRED
        + 2.1635071398 aux_cap_helper__b4,r11,band_HIGH_PAIRED__condX2
        + 1.3908260184 aux_cap_helper__b4,r11,band_HIGH_PAIRED__condX3
        + 1.5453622427 aux_cap_helper__b4,r11,band_HIGH_PAIRED__condX1
        - 0.6181448971 aux_cap_helper__b4,r11,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b4,r11,band_HIGH_PAIRED__Z3  = 0
 c264:  aux_cap_helper__b7,r7,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b7,r7,band_HIGH_PAIRED__Z2 <= 0
 c265:  - aux_cap_b9,r12,band_HIGH_PAIRED
        + 1.3908260184 aux_cap_helper__b9,r12,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b9,r12,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b9,r12,band_HIGH_PAIRED__condX1
        - 0.6181448971 aux_cap_helper__b9,r12,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b9,r12,band_HIGH_PAIRED__Z3  = 0
 c266:  - aux_cap_helper__b5,r6,band_LOW_PAIRED__condX1 <= 0
 c267:  sv_function_helpervar_b6,r13__Z3 + sv_function_helpervar_b6,r13__Z2
        + sv_function_helpervar_b6,r13__Z1  = 1
 c268:  aux_cap_helper__b0,r1,band_LOW_PAIRED__Z1
        + aux_cap_helper__b0,r1,band_LOW_PAIRED__Z2  = 1
 c269:  aux_cap_helper__b3,r4,band_LOW_PAIRED__Z1
        + aux_cap_helper__b3,r4,band_LOW_PAIRED__Z2  = 1
 c270:  aux_cap_helper__b9,r8,band_UNPAIRED__condX2
        + aux_cap_helper__b9,r8,band_UNPAIRED__condX1 - X__b9,r8,band_UNPAIRED
         = 0
 c271:  aux_cap_helper__b1,r6,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b1,r6,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b1,r6,band_HIGH_PAIRED__Z1  = 1
 c272:  - sv_function_helpervar_b9,r7__condX3
        + 7.41632481705671 sv_function_helpervar_b9,r7__Z3 <= 0
 c273:  - aux_cap_helper__b4,r0,band_LOW_PAIRED__condX1 <= 0
 c274:  - 3 aux_cap_helper__b4,r4,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b4,r4,band_HIGH_PAIRED__condX3 <= 0
 c275:  sv_function_helpervar_b6,r4__Z2 + sv_function_helpervar_b6,r4__Z1
        + sv_function_helpervar_b6,r4__Z3  = 1
 c276:  - 2 aux_cap_helper__b4,r7,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b4,r7,band_HIGH_PAIRED__condX2 <= 0
 c277:  - aux_cap_helper__b0,r2,band_UNPAIRED__condX2
        + aux_cap_helper__b0,r2,band_UNPAIRED__Z2 <= 0
 c278:  - aux_cap_helper__b5,r12,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b5,r12,band_HIGH_PAIRED__Z2 <= 0
 c279:  - aux_cap_helper__b6,r4,band_HIGH_PAIRED__condX1 <= 0
 c280:  - aux_cap_b9,r13,band_HIGH_PAIRED
        + 0.927217345653617 aux_cap_helper__b9,r13,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b9,r13,band_HIGH_PAIRED__Z2
        + 1.5453622427 aux_cap_helper__b9,r13,band_HIGH_PAIRED__condX1
        + 2.1635071398 aux_cap_helper__b9,r13,band_HIGH_PAIRED__condX2
        + 1.3908260184 aux_cap_helper__b9,r13,band_HIGH_PAIRED__condX3  = 0
 c281:  sv_function_helpervar_b3,r7__condX2
        + sv_function_helpervar_b3,r7__condX1
        + sv_function_helpervar_b3,r7__condX3 - aux_c_b3,r7  = 0
 c282:  5.2303475434 aux_cap_helper__b6,r5,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b6,r5,band_LOW_PAIRED__condX1
        - 1.4943850124 aux_cap_helper__b6,r5,band_LOW_PAIRED__Z2
        - aux_cap_b6,r5,band_LOW_PAIRED  = 0
 c283:  aux_cap_helper__b1,r2,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b1,r2,band_HIGH_PAIRED__Z3 <= 0
 c284:  - aux_cap_helper__b0,r5,band_HIGH_PAIRED__condX1 <= 0
 c285:  sv_function_helpervar_b7,r12__condX1
        - 0.159946187505732 sv_function_helpervar_b7,r12__Z1 <= 0
 c286:  - aux_cap_helper__b4,r13,band_HIGH_PAIRED__condX1 <= 0
 c287:  aux_cap_helper__b7,r7,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b7,r7,band_HIGH_PAIRED__Z1 <= 0
 c288:  sv_function_helpervar_b5,r9__condX3
        - 15.9946187432626 sv_function_helpervar_b5,r9__Z3 <= 0
 c289:  - sv_function_helpervar_b7,r7__condX2
        + 0.159946187465588 sv_function_helpervar_b7,r7__Z2 <= 0
 c290:  sv_function_helpervar_b5,r11__Z3 + sv_function_helpervar_b5,r11__Z2
        + sv_function_helpervar_b5,r11__Z1  = 1
 c291:  - v__9 + 0.9587 GlobalBidder_PSI__i9,k2
        + 0.98967 GlobalBidder_PSI__i9,k1 + GlobalBidder_PSI__i9,k0  = 0
 c292:  - aux_cap_helper__b1,r12,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b1,r12,band_HIGH_PAIRED__condX1 <= 0
 c293:  - sv_function_helpervar_b9,r5__condX1 <= 0
 c294:  0.927217345653617 aux_cap_helper__b4,r13,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b4,r13,band_HIGH_PAIRED__Z2
        + 1.5453622427 aux_cap_helper__b4,r13,band_HIGH_PAIRED__condX1
        + 1.3908260184 aux_cap_helper__b4,r13,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b4,r13,band_HIGH_PAIRED__condX2
        - aux_cap_b4,r13,band_HIGH_PAIRED  = 0
 c295:  - aux_c_b5,r12 + aux_cap_b5,r12,band_HIGH_PAIRED
        + aux_cap_b5,r12,band_UNPAIRED + aux_cap_b5,r12,band_LOW_PAIRED  = 0
 c296:  aux_cap_helper__b6,r12,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b6,r12,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b6,r12,band_HIGH_PAIRED__Z2  = 1
 c297:  - aux_cap_helper__b1,r2,band_LOW_PAIRED__condX2
        + aux_cap_helper__b1,r2,band_LOW_PAIRED__Z2 <= 0
 c298:  aux_cap_helper__b2,r4,band_UNPAIRED__condX1
        - aux_cap_helper__b2,r4,band_UNPAIRED__Z1 <= 0
 c299:  sv_function_helpervar_b3,r5__Z3 + sv_function_helpervar_b3,r5__Z2
        + sv_function_helpervar_b3,r5__Z1  = 1
 c300:  5.2303475434 aux_cap_helper__b6,r0,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b6,r0,band_LOW_PAIRED__condX1
        - 1.4943850124 aux_cap_helper__b6,r0,band_LOW_PAIRED__Z2
        - aux_cap_b6,r0,band_LOW_PAIRED  = 0
 c301:  X__b8,r3,band_HIGH_PAIRED + X__b8,r3,band_UNPAIRED
        + X__b8,r3,band_LOW_PAIRED - GlobalBidder_W_ir__b8,r3 >= 0
 c302:  sv_function_helpervar_b4,r1__Z2 + sv_function_helpervar_b4,r1__Z3
        + sv_function_helpervar_b4,r1__Z1  = 1
 c303:  aux_cap_helper__b8,r13,band_LOW_PAIRED__Z2
        - aux_cap_helper__b8,r13,band_LOW_PAIRED__condX2 <= 0
 c304:  aux_cap_helper__b0,r2,band_LOW_PAIRED__Z1
        + aux_cap_helper__b0,r2,band_LOW_PAIRED__Z2  = 1
 c305:  - 2 aux_cap_helper__b7,r1,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b7,r1,band_HIGH_PAIRED__condX2 <= 0
 c306:  sv_function_helpervar_b2,r8__condX1
        - 0.159946187457799 sv_function_helpervar_b2,r8__Z1 <= 0
 c307:  - sv_function_helpervar_b9,r12__condX1 <= 0
 c308:  - aux_cap_helper__b9,r0,band_UNPAIRED__condX2
        + aux_cap_helper__b9,r0,band_UNPAIRED__Z2 <= 0
 c309:  0.0102040816326531 X__b7,r10,band_HIGH_PAIRED
        + 0.0102040816326531 X__b7,r10,band_UNPAIRED
        + 0.0102040816326531 X__b7,r10,band_LOW_PAIRED
        - GlobalBidder_W_ir__b7,r10 <= 0
 c310:  - aux_cap_helper__b4,r13,band_UNPAIRED__Z1
        + aux_cap_helper__b4,r13,band_UNPAIRED__condX1 <= 0
 c311:  aux_cap_helper__b3,r5,band_LOW_PAIRED__Z1
        + aux_cap_helper__b3,r5,band_LOW_PAIRED__Z2  = 1
 c312:  - aux_cap_helper__b9,r6,band_LOW_PAIRED__Z1
        + aux_cap_helper__b9,r6,band_LOW_PAIRED__condX1 <= 0
 c313:  - aux_cap_helper__b4,r5,band_UNPAIRED__Z1
        + aux_cap_helper__b4,r5,band_UNPAIRED__condX1 <= 0
 c314:  aux_cap_helper__b5,r8,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b5,r8,band_HIGH_PAIRED__condX2 <= 0
 c315:  - aux_cap_helper__b4,r6,band_LOW_PAIRED__condX1 <= 0
 c316:  - aux_Omega_b3,r11 + 1914.35129815375 aux_sv_b3,r11  = 0
 c317:  - aux_c_b6,r3 + sv_function_helpervar_b6,r3__condX2
        + sv_function_helpervar_b6,r3__condX3
        + sv_function_helpervar_b6,r3__condX1  = 0
 c318:  sv_function_helpervar_b2,r0__condX2
        - 6.18862644261808 sv_function_helpervar_b2,r0__Z2 <= 0
 c319:  sv_function_helpervar_b2,r10__condX3
        + sv_function_helpervar_b2,r10__condX2
        + sv_function_helpervar_b2,r10__condX1 - aux_c_b2,r10  = 0
 c320:  aux_cap_helper__b4,r13,band_LOW_PAIRED__condX1
        - aux_cap_helper__b4,r13,band_LOW_PAIRED__Z1 <= 0
 c321:  - sv_function_helpervar_b7,r9__condX1 <= 0
 c322:  - 2 aux_cap_helper__b8,r12,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b8,r12,band_HIGH_PAIRED__condX2 <= 0
 c323:  - aux_sv_b0,r8 + 36.4816331098689 sv_function_helpervar_b0,r8__Z3
        + 16.7763859206281 sv_function_helpervar_b0,r8__Z2
        + 109.8706625911 sv_function_helpervar_b0,r8__condX1
        + 4.9829738829 sv_function_helpervar_b0,r8__condX2
        + 1.7884146097 sv_function_helpervar_b0,r8__condX3  = 0
 c324:  - 2 aux_cap_helper__b3,r7,band_LOW_PAIRED__Z2
        + aux_cap_helper__b3,r7,band_LOW_PAIRED__condX2 <= 0
 c325:  - 2 aux_cap_helper__b9,r6,band_LOW_PAIRED__Z2
        + aux_cap_helper__b9,r6,band_LOW_PAIRED__condX2 <= 0
 c326:  aux_cap_helper__b2,r7,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b2,r7,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b2,r7,band_HIGH_PAIRED__condX3
        - X__b2,r7,band_HIGH_PAIRED  = 0
 c327:  - 2 aux_cap_helper__b4,r5,band_UNPAIRED__Z2
        + aux_cap_helper__b4,r5,band_UNPAIRED__condX2 <= 0
 c328:  aux_cap_helper__b8,r4,band_UNPAIRED__condX1
        - aux_cap_helper__b8,r4,band_UNPAIRED__Z1 <= 0
 c329:  - 15.9946187432626 sv_function_helpervar_b8,r3__Z3
        + sv_function_helpervar_b8,r3__condX3 <= 0
 c330:  6.71456150061446 sv_function_helpervar_b7,r10__Z3
        - sv_function_helpervar_b7,r10__condX3 <= 0
 c331:  sv_function_helpervar_b2,r9__Z2 + sv_function_helpervar_b2,r9__Z3
        + sv_function_helpervar_b2,r9__Z1  = 1
 c332:  - X__b4,r3,band_UNPAIRED + aux_cap_helper__b4,r3,band_UNPAIRED__condX1
        + aux_cap_helper__b4,r3,band_UNPAIRED__condX2  = 0
 c333:  sv_function_helpervar_b1,r12__condX1
        - 0.15994618744495 sv_function_helpervar_b1,r12__Z1 <= 0
 c334:  - aux_cap_helper__b7,r1,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b7,r1,band_HIGH_PAIRED__condX1 <= 0
 c335:  aux_cap_helper__b4,r8,band_LOW_PAIRED__condX2
        + aux_cap_helper__b4,r8,band_LOW_PAIRED__condX1
        - X__b4,r8,band_LOW_PAIRED  = 0
 c336:  - 2 aux_cap_helper__b1,r12,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b1,r12,band_HIGH_PAIRED__condX2 <= 0
 c337:  aux_cap_helper__b6,r8,band_LOW_PAIRED__Z1
        + aux_cap_helper__b6,r8,band_LOW_PAIRED__Z2  = 1
 c338:  - aux_cap_helper__b5,r1,band_UNPAIRED__condX2
        + aux_cap_helper__b5,r1,band_UNPAIRED__Z2 <= 0
 c339:  - 3 aux_cap_helper__b1,r8,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b1,r8,band_HIGH_PAIRED__condX3 <= 0
 c340:  - 2 aux_cap_helper__b1,r10,band_UNPAIRED__Z2
        + aux_cap_helper__b1,r10,band_UNPAIRED__condX2 <= 0
 c341:  2 aux_cap_helper__b5,r7,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b5,r7,band_HIGH_PAIRED__condX3 <= 0
 c342:  17.4855875792 sv_function_helpervar_b6,r11__condX3
        + 1059.4718673585 sv_function_helpervar_b6,r11__condX1
        + 46.9950588361 sv_function_helpervar_b6,r11__condX2 - aux_sv_b6,r11
        + 347.94871506655 sv_function_helpervar_b6,r11__Z3
        + 161.941805421739 sv_function_helpervar_b6,r11__Z2  = 0
 c343:  - aux_cap_helper__b3,r12,band_LOW_PAIRED__condX2
        + aux_cap_helper__b3,r12,band_LOW_PAIRED__Z2 <= 0
 c344:  aux_cap_helper__b2,r4,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b2,r4,band_UNPAIRED__Z2 <= 0
 c345:  1920.50367357497 aux_sv_b2,r11 - aux_Omega_b2,r11  = 0
 c346:  - 6.67209283899866 sv_function_helpervar_b7,r9__Z2
        + sv_function_helpervar_b7,r9__condX2 <= 0
 c347:  - aux_cap_helper__b8,r12,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b8,r12,band_HIGH_PAIRED__condX1 <= 0
 c348:  aux_cap_helper__b0,r6,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b0,r6,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b0,r6,band_HIGH_PAIRED__condX2
        - X__b0,r6,band_HIGH_PAIRED  = 0
 c349:  7.02419321281872 sv_function_helpervar_b9,r12__Z3
        - sv_function_helpervar_b9,r12__condX3 <= 0
 c350:  - aux_cap_helper__b6,r7,band_LOW_PAIRED__condX2
        + aux_cap_helper__b6,r7,band_LOW_PAIRED__Z2 <= 0
 c351:  - 15.9946187432626 sv_function_helpervar_b4,r12__Z3
        + sv_function_helpervar_b4,r12__condX3 <= 0
 c352:  - aux_cap_helper__b5,r2,band_UNPAIRED__condX2
        + aux_cap_helper__b5,r2,band_UNPAIRED__Z2 <= 0
 c353:  sv_function_helpervar_b9,r10__condX2
        - 7.70903696793851 sv_function_helpervar_b9,r10__Z2 <= 0
 c354:  - sv_function_helpervar_b3,r4__condX1 <= 0
 c355:  - aux_cap_helper__b0,r5,band_LOW_PAIRED__Z1
        + aux_cap_helper__b0,r5,band_LOW_PAIRED__condX1 <= 0
 c356:  aux_cap_helper__b8,r3,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b8,r3,band_UNPAIRED__Z2 <= 0
 c357:  sv_function_helpervar_b0,r2__Z2 + sv_function_helpervar_b0,r2__Z1
        + sv_function_helpervar_b0,r2__Z3  = 1
 c358:  aux_cap_helper__b6,r1,band_UNPAIRED__Z1
        + aux_cap_helper__b6,r1,band_UNPAIRED__Z2  = 1
 c359:  aux_cap_helper__b2,r13,band_LOW_PAIRED__Z2
        - aux_cap_helper__b2,r13,band_LOW_PAIRED__condX2 <= 0
 c360:  aux_cap_helper__b1,r9,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b1,r9,band_HIGH_PAIRED__Z3 <= 0
 c361:  - X__b7,r4,band_UNPAIRED + aux_cap_helper__b7,r4,band_UNPAIRED__condX1
        + aux_cap_helper__b7,r4,band_UNPAIRED__condX2  = 0
 c362:  aux_cap_helper__b3,r7,band_UNPAIRED__Z1
        + aux_cap_helper__b3,r7,band_UNPAIRED__Z2  = 1
 c363:  aux_cap_helper__b1,r9,band_LOW_PAIRED__condX1
        + aux_cap_helper__b1,r9,band_LOW_PAIRED__condX2
        - X__b1,r9,band_LOW_PAIRED  = 0
 c364:  aux_cap_helper__b2,r10,band_LOW_PAIRED__Z1
        + aux_cap_helper__b2,r10,band_LOW_PAIRED__Z2  = 1
 c365:  - 2 aux_cap_helper__b0,r5,band_LOW_PAIRED__Z2
        + aux_cap_helper__b0,r5,band_LOW_PAIRED__condX2 <= 0
 c366:  - aux_cap_helper__b8,r13,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b8,r13,band_HIGH_PAIRED__condX1 <= 0
 c367:  sv_function_helpervar_b6,r7__condX1
        - 0.159946187457337 sv_function_helpervar_b6,r7__Z1 <= 0
 c368:  - aux_cap_helper__b4,r6,band_UNPAIRED__condX1 <= 0
 c369:  - 3 aux_cap_helper__b7,r2,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b7,r2,band_HIGH_PAIRED__condX3 <= 0
 c370:  - aux_cap_helper__b7,r7,band_UNPAIRED__condX1 <= 0
 c371:  sv_function_helpervar_b9,r10__condX1
        + sv_function_helpervar_b9,r10__condX2
        + sv_function_helpervar_b9,r10__condX3 - aux_c_b9,r10  = 0
 c372:  aux_cap_helper__b5,r7,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b5,r7,band_HIGH_PAIRED__condX2 <= 0
 c373:  - aux_sv_b2,r11 + 53.304833033958 sv_function_helpervar_b2,r11__Z3
        + 24.5630227856076 sv_function_helpervar_b2,r11__Z2
        + 2.6242444484 sv_function_helpervar_b2,r11__condX3
        + 7.2662268524 sv_function_helpervar_b2,r11__condX2
        + 160.8367693657 sv_function_helpervar_b2,r11__condX1  = 0
 c374:  - aux_cap_helper__b1,r8,band_UNPAIRED__condX1 <= 0
 c375:  aux_cap_helper__b6,r0,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b6,r0,band_LOW_PAIRED__Z2 <= 0
 c376:  6.02685562864558 sv_function_helpervar_b5,r6__Z3
        - sv_function_helpervar_b5,r6__condX3 <= 0
 c377:  0.1599461874636 sv_function_helpervar_b1,r1__Z2
        - sv_function_helpervar_b1,r1__condX2 <= 0
 c378:  - aux_Omega_b9,r5 + 1246.75088198607 aux_sv_b9,r5  = 0
 c379:  0.159946187464243 sv_function_helpervar_b5,r11__Z2
        - sv_function_helpervar_b5,r11__condX2 <= 0
 c380:  - sv_function_helpervar_b8,r10__condX3
        + 7.6310207645995 sv_function_helpervar_b8,r10__Z3 <= 0
 c381:  - 2 aux_cap_helper__b4,r5,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b4,r5,band_HIGH_PAIRED__condX2 <= 0
 c382:  sv_function_helpervar_b5,r5__condX1
        - 0.159946187436384 sv_function_helpervar_b5,r5__Z1 <= 0
 c383:  - 0.159946187447607 sv_function_helpervar_b0,r4__Z1
        + sv_function_helpervar_b0,r4__condX1 <= 0
 c384:  - aux_cap_helper__b9,r10,band_UNPAIRED__condX2
        + aux_cap_helper__b9,r10,band_UNPAIRED__Z2 <= 0
 c385:  5.2303475434 aux_cap_helper__b1,r2,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b1,r2,band_LOW_PAIRED__condX1
        - 1.4943850124 aux_cap_helper__b1,r2,band_LOW_PAIRED__Z2
        - aux_cap_b1,r2,band_LOW_PAIRED  = 0
 c386:  - aux_cap_b1,r7,band_UNPAIRED
        + 0.8035888618 aux_cap_helper__b1,r7,band_UNPAIRED__condX1
        + 1.1250244065 aux_cap_helper__b1,r7,band_UNPAIRED__condX2
        - 0.3214355447 aux_cap_helper__b1,r7,band_UNPAIRED__Z2  = 0
 c387:  - aux_Omega_b5,r7 + 1300.36546851094 aux_sv_b5,r7  = 0
 c388:  - sv_function_helpervar_b4,r13__condX1 <= 0
 c389:  - aux_cap_b5,r13,band_LOW_PAIRED
        + 5.2303475434 aux_cap_helper__b5,r13,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b5,r13,band_LOW_PAIRED__condX1
        - 1.4943850124 aux_cap_helper__b5,r13,band_LOW_PAIRED__Z2  = 0
 c390:  - 15.9946187432626 sv_function_helpervar_b4,r1__Z3
        + sv_function_helpervar_b4,r1__condX3 <= 0
 c391:  - sv_function_helpervar_b4,r2__condX1 <= 0
 c392:  aux_cap_helper__b8,r3,band_UNPAIRED__condX1
        - aux_cap_helper__b8,r3,band_UNPAIRED__Z1 <= 0
 c393:  6.13038020413258 sv_function_helpervar_b1,r10__Z3
        - sv_function_helpervar_b1,r10__condX3 <= 0
 c394:  aux_cap_helper__b6,r10,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b6,r10,band_HIGH_PAIRED__Z2 <= 0
 c395:  aux_cap_b0,r11,band_UNPAIRED - aux_c_b0,r11
        + aux_cap_b0,r11,band_HIGH_PAIRED + aux_cap_b0,r11,band_LOW_PAIRED  = 0
 c396:  - aux_cap_helper__b9,r11,band_UNPAIRED__Z1
        + aux_cap_helper__b9,r11,band_UNPAIRED__condX1 <= 0
 c397:  sv_function_helpervar_b4,r6__condX3
        + sv_function_helpervar_b4,r6__condX1
        + sv_function_helpervar_b4,r6__condX2 - aux_c_b4,r6  = 0
 c398:  - aux_cap_b8,r11,band_UNPAIRED
        + 1.1250244065 aux_cap_helper__b8,r11,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b8,r11,band_UNPAIRED__condX1
        - 0.3214355447 aux_cap_helper__b8,r11,band_UNPAIRED__Z2  = 0
 c399:  - aux_cap_b7,r0,band_UNPAIRED
        - 0.3214355447 aux_cap_helper__b7,r0,band_UNPAIRED__Z2
        + 1.1250244065 aux_cap_helper__b7,r0,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b7,r0,band_UNPAIRED__condX1  = 0
 c400:  aux_cap_helper__b5,r4,band_UNPAIRED__Z1
        + aux_cap_helper__b5,r4,band_UNPAIRED__Z2  = 1
 c401:  - X__b9,r1,band_UNPAIRED + aux_cap_helper__b9,r1,band_UNPAIRED__condX1
        + aux_cap_helper__b9,r1,band_UNPAIRED__condX2  = 0
 c402:  - aux_cap_helper__b5,r3,band_LOW_PAIRED__condX1 <= 0
 c403:  aux_cap_helper__b4,r1,band_LOW_PAIRED__condX1
        - aux_cap_helper__b4,r1,band_LOW_PAIRED__Z1 <= 0
 c404:  2 aux_cap_helper__b5,r8,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b5,r8,band_HIGH_PAIRED__condX3 <= 0
 c405:  - 2 aux_cap_helper__b9,r11,band_UNPAIRED__Z2
        + aux_cap_helper__b9,r11,band_UNPAIRED__condX2 <= 0
 c406:  aux_cap_helper__b2,r3,band_UNPAIRED__Z2
        - aux_cap_helper__b2,r3,band_UNPAIRED__condX2 <= 0
 c407:  aux_cap_helper__b8,r8,band_LOW_PAIRED__condX1
        + aux_cap_helper__b8,r8,band_LOW_PAIRED__condX2
        - X__b8,r8,band_LOW_PAIRED  = 0
 c408:  - aux_sv_b5,r12 + 177.608257672181 sv_function_helpervar_b5,r12__Z2
        + 406.488111760978 sv_function_helpervar_b5,r12__Z3
        + 17.904369337 sv_function_helpervar_b5,r12__condX3
        + 59.172364328 sv_function_helpervar_b5,r12__condX2
        + 1169.5974422991 sv_function_helpervar_b5,r12__condX1  = 0
 c409:  sv_function_helpervar_b6,r12__condX3
        - 15.9946187432626 sv_function_helpervar_b6,r12__Z3 <= 0
 c410:  aux_cap_helper__b4,r1,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b4,r1,band_LOW_PAIRED__Z2 <= 0
 c411:  - aux_cap_helper__b4,r5,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b4,r5,band_HIGH_PAIRED__condX1 <= 0
 c412:  5.47156313732081 sv_function_helpervar_b6,r4__Z3
        - sv_function_helpervar_b6,r4__condX3 <= 0
 c413:  - aux_Omega_b5,r5 + 650.750566999416 aux_sv_b5,r5  = 0
 c414:  - aux_Omega_b9,r1 + 437.284803413729 aux_sv_b9,r1  = 0
 c415:  - aux_cap_helper__b6,r3,band_HIGH_PAIRED__condX1 <= 0
 c416:  aux_cap_helper__b2,r11,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b2,r11,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b2,r11,band_HIGH_PAIRED__Z1  = 1
 c417:  sv_function_helpervar_b3,r3__condX3
        - 15.9946187432626 sv_function_helpervar_b3,r3__Z3 <= 0
 c418:  - 2 aux_cap_helper__b8,r13,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b8,r13,band_HIGH_PAIRED__condX2 <= 0
 c419:  - 15.9946187432626 sv_function_helpervar_b9,r1__Z3
        + sv_function_helpervar_b9,r1__condX3 <= 0
 c420:  - 0.159946187436826 sv_function_helpervar_b4,r3__Z1
        + sv_function_helpervar_b4,r3__condX1 <= 0
 c421:  X__b9,r9,band_LOW_PAIRED + X__b9,r9,band_UNPAIRED
        + X__b9,r9,band_HIGH_PAIRED - GlobalBidder_W_ir__b9,r9 >= 0
 c422:  aux_cap_helper__b4,r1,band_LOW_PAIRED__condX2
        + aux_cap_helper__b4,r1,band_LOW_PAIRED__condX1
        - X__b4,r1,band_LOW_PAIRED  = 0
 c423:  19.0879364968027 sv_function_helpervar_b1,r5__Z2
        + 43.3078609917909 sv_function_helpervar_b1,r5__Z3
        + 1.9428515363 sv_function_helpervar_b1,r5__condX3
        + 6.2238542689 sv_function_helpervar_b1,r5__condX2
        + 125.5635946637 sv_function_helpervar_b1,r5__condX1 - aux_sv_b1,r5
         = 0
 c424:  aux_cap_helper__b6,r10,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b6,r10,band_HIGH_PAIRED__Z1 <= 0
 c425:  - aux_cap_b9,r9,band_LOW_PAIRED
        + 3.735962531 aux_cap_helper__b9,r9,band_LOW_PAIRED__condX1
        + 5.2303475434 aux_cap_helper__b9,r9,band_LOW_PAIRED__condX2
        - 1.4943850124 aux_cap_helper__b9,r9,band_LOW_PAIRED__Z2  = 0
 c426:  - 0.159946187520236 sv_function_helpervar_b7,r0__Z1
        + sv_function_helpervar_b7,r0__condX1 <= 0
 c427:  aux_cap_helper__b6,r12,band_LOW_PAIRED__Z2
        + aux_cap_helper__b6,r12,band_LOW_PAIRED__Z1  = 1
 c428:  X__b9,r1,band_UNPAIRED + X__b9,r1,band_LOW_PAIRED
        + X__b9,r1,band_HIGH_PAIRED - GlobalBidder_W_ir__b9,r1 >= 0
 c429:  aux_Omega_b7,r10 + aux_Omega_b7,r11 + aux_Omega_b7,r12
        + aux_Omega_b7,r13 + aux_Omega_b7,r0 + aux_Omega_b7,r1
        + aux_Omega_b7,r2 + aux_Omega_b7,r3 + aux_Omega_b7,r4 + aux_Omega_b7,r5
        + aux_Omega_b7,r6 + aux_Omega_b7,r7 + aux_Omega_b7,r8 + aux_Omega_b7,r9
        - GlobalBidder_PSI__i7,k0 - 18497526.8940961 GlobalBidder_hat{W}__i7,k0
        >= -18497526.8940961
 c430:  - aux_cap_helper__b1,r7,band_UNPAIRED__condX1 <= 0
 c431:  - 6.12210391322581 sv_function_helpervar_b0,r7__Z2
        + sv_function_helpervar_b0,r7__condX2 <= 0
 c432:  - aux_cap_helper__b1,r5,band_LOW_PAIRED__condX2
        + aux_cap_helper__b1,r5,band_LOW_PAIRED__Z2 <= 0
 c433:  aux_cap_helper__b4,r6,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b4,r6,band_HIGH_PAIRED__Z1 <= 0
 c434:  - aux_cap_helper__b7,r4,band_LOW_PAIRED__condX2
        + aux_cap_helper__b7,r4,band_LOW_PAIRED__Z2 <= 0
 c435:  X__b4,r13,band_LOW_PAIRED + X__b5,r13,band_LOW_PAIRED
        + X__b8,r13,band_LOW_PAIRED + X__b7,r13,band_LOW_PAIRED
        + X__b6,r13,band_LOW_PAIRED + X__b9,r13,band_LOW_PAIRED
        + X__b2,r13,band_LOW_PAIRED + X__b3,r13,band_LOW_PAIRED
        + X__b1,r13,band_LOW_PAIRED + X__b0,r13,band_LOW_PAIRED <= 2
 c436:  - 6.50772434820002 sv_function_helpervar_b0,r0__Z2
        + sv_function_helpervar_b0,r0__condX2 <= 0
 c437:  - aux_cap_helper__b8,r7,band_UNPAIRED__condX1 <= 0
 c438:  - aux_cap_helper__b0,r8,band_UNPAIRED__condX1 <= 0
 c439:  aux_cap_b1,r4,band_LOW_PAIRED + aux_cap_b1,r4,band_UNPAIRED
        + aux_cap_b1,r4,band_HIGH_PAIRED - aux_c_b1,r4  = 0
 c440:  aux_cap_helper__b6,r1,band_UNPAIRED__Z2
        - aux_cap_helper__b6,r1,band_UNPAIRED__condX2 <= 0
 c441:  - aux_cap_helper__b4,r0,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b4,r0,band_HIGH_PAIRED__Z3 <= 0
 c442:  aux_cap_helper__b7,r4,band_UNPAIRED__condX1
        - aux_cap_helper__b7,r4,band_UNPAIRED__Z1 <= 0
 c443:  aux_cap_helper__b4,r6,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b4,r6,band_HIGH_PAIRED__Z2 <= 0
 c444:  aux_cap_b2,r6,band_HIGH_PAIRED - aux_c_b2,r6
        + aux_cap_b2,r6,band_UNPAIRED + aux_cap_b2,r6,band_LOW_PAIRED  = 0
 c445:  aux_cap_b1,r8,band_UNPAIRED + aux_cap_b1,r8,band_HIGH_PAIRED
        + aux_cap_b1,r8,band_LOW_PAIRED - aux_c_b1,r8  = 0
 c446:  aux_cap_helper__b7,r4,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b7,r4,band_UNPAIRED__Z2 <= 0
 c447:  X__b6,r0,band_LOW_PAIRED + X__b8,r0,band_LOW_PAIRED
        + X__b1,r0,band_LOW_PAIRED + X__b2,r0,band_LOW_PAIRED
        + X__b3,r0,band_LOW_PAIRED + X__b5,r0,band_LOW_PAIRED
        + X__b4,r0,band_LOW_PAIRED + X__b7,r0,band_LOW_PAIRED
        + X__b9,r0,band_LOW_PAIRED + X__b0,r0,band_LOW_PAIRED <= 2
 c448:  - X__b9,r7,band_LOW_PAIRED
        + aux_cap_helper__b9,r7,band_LOW_PAIRED__condX1
        + aux_cap_helper__b9,r7,band_LOW_PAIRED__condX2  = 0
 c449:  aux_cap_helper__b4,r3,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b4,r3,band_HIGH_PAIRED__Z3 <= 0
 c450:  aux_cap_helper__b5,r13,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b5,r13,band_HIGH_PAIRED__condX2 <= 0
 c451:  aux_cap_helper__b8,r10,band_LOW_PAIRED__Z2
        - aux_cap_helper__b8,r10,band_LOW_PAIRED__condX2 <= 0
 c452:  aux_cap_helper__b3,r5,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b3,r5,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b3,r5,band_HIGH_PAIRED__condX3
        - X__b3,r5,band_HIGH_PAIRED  = 0
 c453:  - aux_cap_helper__b3,r9,band_HIGH_PAIRED__condX1 <= 0
 c454:  aux_cap_helper__b4,r10,band_UNPAIRED__Z2
        - aux_cap_helper__b4,r10,band_UNPAIRED__condX2 <= 0
 c455:  aux_cap_helper__b5,r5,band_UNPAIRED__condX1
        - aux_cap_helper__b5,r5,band_UNPAIRED__Z1 <= 0
 c456:  - sv_function_helpervar_b8,r10__condX2
        + 0.159946187439248 sv_function_helpervar_b8,r10__Z2 <= 0
 c457:  - 2 aux_cap_helper__b9,r3,band_UNPAIRED__Z2
        + aux_cap_helper__b9,r3,band_UNPAIRED__condX2 <= 0
 c458:  sv_function_helpervar_b2,r4__condX3
        - 15.9946187432626 sv_function_helpervar_b2,r4__Z3 <= 0
 c459:  aux_cap_helper__b1,r0,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b1,r0,band_HIGH_PAIRED__condX2 <= 0
 c460:  aux_cap_helper__b7,r13,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b7,r13,band_UNPAIRED__Z2 <= 0
 c461:  - aux_cap_b8,r6,band_LOW_PAIRED
        + 5.2303475434 aux_cap_helper__b8,r6,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b8,r6,band_LOW_PAIRED__condX1
        - 1.4943850124 aux_cap_helper__b8,r6,band_LOW_PAIRED__Z2  = 0
 c462:  aux_cap_b0,r6,band_HIGH_PAIRED - aux_c_b0,r6
        + aux_cap_b0,r6,band_LOW_PAIRED + aux_cap_b0,r6,band_UNPAIRED  = 0
 c463:  - aux_cap_helper__b4,r0,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b4,r0,band_HIGH_PAIRED__Z2 <= 0
 c464:  - aux_Omega_b4,r2 + 113.264402450619 aux_sv_b4,r2  = 0
 c465:  1.1250244065 aux_cap_helper__b3,r13,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b3,r13,band_UNPAIRED__condX1
        - 0.3214355447 aux_cap_helper__b3,r13,band_UNPAIRED__Z2
        - aux_cap_b3,r13,band_UNPAIRED  = 0
 c466:  aux_cap_helper__b6,r11,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b6,r11,band_UNPAIRED__Z2 <= 0
 c467:  aux_cap_helper__b4,r0,band_LOW_PAIRED__Z1
        + aux_cap_helper__b4,r0,band_LOW_PAIRED__Z2  = 1
 c468:  - aux_cap_helper__b3,r6,band_UNPAIRED__condX1 <= 0
 c469:  aux_cap_helper__b7,r6,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b7,r6,band_HIGH_PAIRED__Z2 <= 0
 c470:  GlobalBidder_PSI__i9,k1 - 18497526.8940961 GlobalBidder_hat{W}__i9,k1
        <= 0
 c471:  - aux_cap_helper__b3,r10,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b3,r10,band_HIGH_PAIRED__Z3 <= 0
 c472:  sv_function_helpervar_b5,r1__condX2
        - 5.85899563710341 sv_function_helpervar_b5,r1__Z2 <= 0
 c473:  2 aux_cap_helper__b1,r0,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b1,r0,band_HIGH_PAIRED__condX3 <= 0
 c474:  - sv_function_helpervar_b3,r6__condX1 <= 0
 c475:  - 6.55357470302108 sv_function_helpervar_b8,r5__Z2
        + sv_function_helpervar_b8,r5__condX2 <= 0
 c476:  - 3 aux_cap_helper__b7,r3,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b7,r3,band_HIGH_PAIRED__condX3 <= 0
 c477:  aux_cap_helper__b8,r2,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b8,r2,band_HIGH_PAIRED__condX2 <= 0
 c478:  0.159946187486893 sv_function_helpervar_b5,r2__Z2
        - sv_function_helpervar_b5,r2__condX2 <= 0
 c479:  - sv_function_helpervar_b3,r3__condX3
        + 5.6831725135299 sv_function_helpervar_b3,r3__Z3 <= 0
 c480:  - aux_cap_helper__b5,r0,band_LOW_PAIRED__condX1 <= 0
 c481:  aux_cap_helper__b0,r12,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b0,r12,band_HIGH_PAIRED__condX2 <= 0
 c482:  aux_cap_helper__b1,r6,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b1,r6,band_HIGH_PAIRED__Z1 <= 0
 c483:  1.1250244065 aux_cap_helper__b1,r0,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b1,r0,band_UNPAIRED__condX1
        - aux_cap_b1,r0,band_UNPAIRED
        - 0.3214355447 aux_cap_helper__b1,r0,band_UNPAIRED__Z2  = 0
 c484:  2 aux_cap_helper__b8,r2,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b8,r2,band_HIGH_PAIRED__condX3 <= 0
 c485:  aux_cap_helper__b8,r0,band_UNPAIRED__Z2
        - aux_cap_helper__b8,r0,band_UNPAIRED__condX2 <= 0
 c486:  aux_cap_helper__b1,r2,band_LOW_PAIRED__condX2
        + aux_cap_helper__b1,r2,band_LOW_PAIRED__condX1
        - X__b1,r2,band_LOW_PAIRED  = 0
 c487:  - sv_function_helpervar_b8,r6__condX1 <= 0
 c488:  - X__b9,r8,band_HIGH_PAIRED
        + aux_cap_helper__b9,r8,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b9,r8,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b9,r8,band_HIGH_PAIRED__condX3  = 0
 c489:  - aux_cap_helper__b9,r3,band_UNPAIRED__Z1
        + aux_cap_helper__b9,r3,band_UNPAIRED__condX1 <= 0
 c490:  - aux_cap_helper__b2,r7,band_UNPAIRED__condX1 <= 0
 c491:  sv_function_helpervar_b7,r0__Z3 + sv_function_helpervar_b7,r0__Z2
        + sv_function_helpervar_b7,r0__Z1  = 1
 c492:  0.159946187500414 sv_function_helpervar_b1,r13__Z2
        - sv_function_helpervar_b1,r13__condX2 <= 0
 c493:  - 15.9946187432626 sv_function_helpervar_b8,r5__Z3
        + sv_function_helpervar_b8,r5__condX3 <= 0
 c494:  aux_cap_helper__b3,r4,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b3,r4,band_LOW_PAIRED__Z2 <= 0
 c495:  - sv_function_helpervar_b9,r7__condX2
        + 0.159946187435082 sv_function_helpervar_b9,r7__Z2 <= 0
 c496:  - aux_Omega_b5,r4 + 963.531953025222 aux_sv_b5,r4  = 0
 c497:  2 aux_cap_helper__b5,r13,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b5,r13,band_HIGH_PAIRED__condX3 <= 0
 c498:  - aux_Omega_b9,r7 + 4989.92262931125 aux_sv_b9,r7  = 0
 c499:  aux_cap_helper__b9,r3,band_LOW_PAIRED__condX1
        - aux_cap_helper__b9,r3,band_LOW_PAIRED__Z1 <= 0
 c500:  aux_cap_helper__b1,r3,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b1,r3,band_HIGH_PAIRED__Z3 <= 0
 c501:  X__b7,r13,band_UNPAIRED + X__b8,r13,band_UNPAIRED
        + X__b4,r13,band_UNPAIRED + X__b2,r13,band_UNPAIRED
        + X__b0,r13,band_UNPAIRED + X__b6,r13,band_UNPAIRED
        + X__b3,r13,band_UNPAIRED + X__b9,r13,band_UNPAIRED
        + X__b1,r13,band_UNPAIRED + X__b5,r13,band_UNPAIRED <= 2
 c502:  aux_cap_helper__b1,r6,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b1,r6,band_HIGH_PAIRED__Z2 <= 0
 c503:  aux_cap_helper__b8,r4,band_UNPAIRED__condX2
        + aux_cap_helper__b8,r4,band_UNPAIRED__condX1 - X__b8,r4,band_UNPAIRED
         = 0
 c504:  aux_cap_helper__b9,r3,band_LOW_PAIRED__condX2
        + aux_cap_helper__b9,r3,band_LOW_PAIRED__condX1
        - X__b9,r3,band_LOW_PAIRED  = 0
 c505:  aux_cap_helper__b6,r11,band_UNPAIRED__condX1
        - aux_cap_helper__b6,r11,band_UNPAIRED__Z1 <= 0
 c506:  - aux_cap_helper__b3,r3,band_UNPAIRED__condX2
        + aux_cap_helper__b3,r3,band_UNPAIRED__Z2 <= 0
 c507:  aux_cap_helper__b7,r6,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b7,r6,band_HIGH_PAIRED__Z1 <= 0
 c508:  - aux_cap_helper__b3,r10,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b3,r10,band_HIGH_PAIRED__Z2 <= 0
 c509:  sv_function_helpervar_b3,r1__condX3
        + sv_function_helpervar_b3,r1__condX1
        + sv_function_helpervar_b3,r1__condX2 - aux_c_b3,r1  = 0
 c510:  - sv_function_helpervar_b8,r0__condX3
        + 6.25776005044761 sv_function_helpervar_b8,r0__Z3 <= 0
 c511:  aux_cap_helper__b9,r13,band_UNPAIRED__condX2
        + aux_cap_helper__b9,r13,band_UNPAIRED__condX1
        - X__b9,r13,band_UNPAIRED  = 0
 c512:  - aux_cap_helper__b4,r2,band_UNPAIRED__condX2
        + aux_cap_helper__b4,r2,band_UNPAIRED__Z2 <= 0
 c513:  aux_cap_helper__b9,r3,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b9,r3,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b9,r3,band_HIGH_PAIRED__Z3  = 1
 c514:  5.73573107720359 sv_function_helpervar_b6,r0__Z3
        - sv_function_helpervar_b6,r0__condX3 <= 0
 c515:  0.927217345653617 aux_cap_helper__b4,r12,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b4,r12,band_HIGH_PAIRED__Z2
        - aux_cap_b4,r12,band_HIGH_PAIRED
        + 1.3908260184 aux_cap_helper__b4,r12,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b4,r12,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b4,r12,band_HIGH_PAIRED__condX1  = 0
 c516:  73.7141404582 sv_function_helpervar_b7,r7__condX2
        + 1850.3233256079 sv_function_helpervar_b7,r7__condX1
        + 32.9035499519 sv_function_helpervar_b7,r7__condX3
        + 284.16186578104 sv_function_helpervar_b7,r7__Z2
        + 569.839379922388 sv_function_helpervar_b7,r7__Z3 - aux_sv_b7,r7  = 0
 c517:  - 15.9946187432626 sv_function_helpervar_b2,r11__Z3
        + sv_function_helpervar_b2,r11__condX3 <= 0
 c518:  - X__b3,r6,band_UNPAIRED + aux_cap_helper__b3,r6,band_UNPAIRED__condX1
        + aux_cap_helper__b3,r6,band_UNPAIRED__condX2  = 0
 c519:  sv_function_helpervar_b8,r6__condX2
        + sv_function_helpervar_b8,r6__condX1
        + sv_function_helpervar_b8,r6__condX3 - aux_c_b8,r6  = 0
 c520:  - aux_cap_helper__b9,r6,band_UNPAIRED__condX1 <= 0
 c521:  - aux_Omega_b4,r3 + 481.217817150869 aux_sv_b4,r3  = 0
 c522:  - aux_cap_helper__b7,r0,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b7,r0,band_HIGH_PAIRED__condX1 <= 0
 c523:  - aux_cap_b1,r7,band_LOW_PAIRED
        + 3.735962531 aux_cap_helper__b1,r7,band_LOW_PAIRED__condX1
        + 5.2303475434 aux_cap_helper__b1,r7,band_LOW_PAIRED__condX2
        - 1.4943850124 aux_cap_helper__b1,r7,band_LOW_PAIRED__Z2  = 0
 c524:  aux_cap_helper__b9,r9,band_UNPAIRED__Z1
        + aux_cap_helper__b9,r9,band_UNPAIRED__Z2  = 1
 c525:  aux_cap_helper__b1,r13,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b1,r13,band_HIGH_PAIRED__Z2 <= 0
 c526:  X__b5,r8,band_UNPAIRED + X__b7,r8,band_UNPAIRED
        + X__b2,r8,band_UNPAIRED + X__b6,r8,band_UNPAIRED
        + X__b1,r8,band_UNPAIRED + X__b0,r8,band_UNPAIRED
        + X__b9,r8,band_UNPAIRED + X__b3,r8,band_UNPAIRED
        + X__b4,r8,band_UNPAIRED + X__b8,r8,band_UNPAIRED <= 2
 c527:  sv_function_helpervar_b6,r9__condX3
        + sv_function_helpervar_b6,r9__condX2
        + sv_function_helpervar_b6,r9__condX1 - aux_c_b6,r9  = 0
 c528:  - 2 aux_cap_helper__b7,r0,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b7,r0,band_HIGH_PAIRED__condX2 <= 0
 c529:  sv_function_helpervar_b2,r10__condX2
        - 6.55138684469449 sv_function_helpervar_b2,r10__Z2 <= 0
 c530:  - aux_cap_helper__b9,r12,band_LOW_PAIRED__condX2
        + aux_cap_helper__b9,r12,band_LOW_PAIRED__Z2 <= 0
 c531:  aux_cap_helper__b6,r3,band_UNPAIRED__Z2
        + aux_cap_helper__b6,r3,band_UNPAIRED__Z1  = 1
 c532:  - 5.65845556019633 sv_function_helpervar_b0,r10__Z2
        + sv_function_helpervar_b0,r10__condX2 <= 0
 c533:  - aux_cap_helper__b1,r10,band_UNPAIRED__Z1
        + aux_cap_helper__b1,r10,band_UNPAIRED__condX1 <= 0
 c534:  1.1250244065 aux_cap_helper__b1,r13,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b1,r13,band_UNPAIRED__condX1
        - 0.3214355447 aux_cap_helper__b1,r13,band_UNPAIRED__Z2
        - aux_cap_b1,r13,band_UNPAIRED  = 0
 c535:  - aux_cap_helper__b0,r4,band_HIGH_PAIRED__condX1 <= 0
 c536:  - sv_function_helpervar_b4,r9__condX2
        + 0.159946187437935 sv_function_helpervar_b4,r9__Z2 <= 0
 c537:  - aux_cap_helper__b5,r10,band_LOW_PAIRED__condX2
        + aux_cap_helper__b5,r10,band_LOW_PAIRED__Z2 <= 0
 c538:  - sv_function_helpervar_b1,r2__condX2
        + 0.159946187459287 sv_function_helpervar_b1,r2__Z2 <= 0
 c539:  aux_cap_helper__b2,r11,band_LOW_PAIRED__Z2
        + aux_cap_helper__b2,r11,band_LOW_PAIRED__Z1  = 1
 c540:  0.8035888618 aux_cap_helper__b2,r13,band_UNPAIRED__condX1
        + 1.1250244065 aux_cap_helper__b2,r13,band_UNPAIRED__condX2
        - 0.3214355447 aux_cap_helper__b2,r13,band_UNPAIRED__Z2
        - aux_cap_b2,r13,band_UNPAIRED  = 0
 c541:  aux_cap_helper__b0,r8,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b0,r8,band_LOW_PAIRED__Z2 <= 0
 c542:  - 0.3214355447 aux_cap_helper__b4,r13,band_UNPAIRED__Z2
        + 1.1250244065 aux_cap_helper__b4,r13,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b4,r13,band_UNPAIRED__condX1
        - aux_cap_b4,r13,band_UNPAIRED  = 0
 c543:  aux_cap_helper__b6,r8,band_LOW_PAIRED__condX1
        + aux_cap_helper__b6,r8,band_LOW_PAIRED__condX2
        - X__b6,r8,band_LOW_PAIRED  = 0
 c544:  sv_function_helpervar_b7,r5__Z1 + sv_function_helpervar_b7,r5__Z2
        + sv_function_helpervar_b7,r5__Z3  = 1
 c545:  - aux_sv_b7,r9 + 283.568037161185 sv_function_helpervar_b7,r9__Z2
        + 588.355229725454 sv_function_helpervar_b7,r9__Z3
        + 1850.3233257304 sv_function_helpervar_b7,r9__condX1
        + 77.4268180161 sv_function_helpervar_b7,r9__condX2
        + 31.7459199952 sv_function_helpervar_b7,r9__condX3  = 0
 c546:  1289.3272635294 sv_function_helpervar_b4,r9__condX1
        + 20.0015801447 sv_function_helpervar_b4,r9__condX3
        + 63.5990857666 sv_function_helpervar_b4,r9__condX2 - aux_sv_b4,r9
        + 443.871166735829 sv_function_helpervar_b4,r9__Z3
        + 196.050548868407 sv_function_helpervar_b4,r9__Z2  = 0
 c547:  - 0.3214355447 aux_cap_helper__b4,r9,band_UNPAIRED__Z2
        - aux_cap_b4,r9,band_UNPAIRED
        + 0.8035888618 aux_cap_helper__b4,r9,band_UNPAIRED__condX1
        + 1.1250244065 aux_cap_helper__b4,r9,band_UNPAIRED__condX2  = 0
 c548:  aux_cap_helper__b2,r10,band_LOW_PAIRED__Z2
        - aux_cap_helper__b2,r10,band_LOW_PAIRED__condX2 <= 0
 c549:  - aux_cap_helper__b3,r11,band_UNPAIRED__condX1 <= 0
 c550:  aux_cap_helper__b1,r13,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b1,r13,band_HIGH_PAIRED__Z1 <= 0
 c551:  aux_cap_helper__b2,r8,band_UNPAIRED__Z1
        + aux_cap_helper__b2,r8,band_UNPAIRED__Z2  = 1
 c552:  - X__b6,r13,band_UNPAIRED
        + aux_cap_helper__b6,r13,band_UNPAIRED__condX1
        + aux_cap_helper__b6,r13,band_UNPAIRED__condX2  = 0
 c553:  - aux_cap_helper__b1,r4,band_UNPAIRED__Z1
        + aux_cap_helper__b1,r4,band_UNPAIRED__condX1 <= 0
 c554:  aux_cap_helper__b4,r11,band_LOW_PAIRED__Z2
        - aux_cap_helper__b4,r11,band_LOW_PAIRED__condX2 <= 0
 c555:  - aux_cap_helper__b8,r9,band_LOW_PAIRED__Z1
        + aux_cap_helper__b8,r9,band_LOW_PAIRED__condX1 <= 0
 c556:  aux_cap_b6,r8,band_LOW_PAIRED + aux_cap_b6,r8,band_UNPAIRED
        - aux_c_b6,r8 + aux_cap_b6,r8,band_HIGH_PAIRED  = 0
 c557:  - aux_c_b6,r10 + sv_function_helpervar_b6,r10__condX3
        + sv_function_helpervar_b6,r10__condX2
        + sv_function_helpervar_b6,r10__condX1  = 0
 c558:  aux_cap_helper__b7,r3,band_LOW_PAIRED__Z1
        + aux_cap_helper__b7,r3,band_LOW_PAIRED__Z2  = 1
 c559:  - aux_c_b7,r6 + aux_cap_b7,r6,band_HIGH_PAIRED
        + aux_cap_b7,r6,band_UNPAIRED + aux_cap_b7,r6,band_LOW_PAIRED  = 0
 c560:  - X__b1,r0,band_UNPAIRED + aux_cap_helper__b1,r0,band_UNPAIRED__condX2
        + aux_cap_helper__b1,r0,band_UNPAIRED__condX1  = 0
 c561:  sv_function_helpervar_b1,r4__condX1
        + sv_function_helpervar_b1,r4__condX2
        + sv_function_helpervar_b1,r4__condX3 - aux_c_b1,r4  = 0
 c562:  aux_cap_helper__b4,r4,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b4,r4,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b4,r4,band_HIGH_PAIRED__Z2  = 1
 c563:  - 2 aux_cap_helper__b4,r13,band_UNPAIRED__Z2
        + aux_cap_helper__b4,r13,band_UNPAIRED__condX2 <= 0
 c564:  - 2 aux_cap_helper__b1,r4,band_UNPAIRED__Z2
        + aux_cap_helper__b1,r4,band_UNPAIRED__condX2 <= 0
 c565:  - X__b1,r1,band_HIGH_PAIRED
        + aux_cap_helper__b1,r1,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b1,r1,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b1,r1,band_HIGH_PAIRED__condX2  = 0
 c566:  - 2 aux_cap_helper__b8,r9,band_LOW_PAIRED__Z2
        + aux_cap_helper__b8,r9,band_LOW_PAIRED__condX2 <= 0
 c567:  aux_cap_helper__b8,r10,band_UNPAIRED__Z2
        + aux_cap_helper__b8,r10,band_UNPAIRED__Z1  = 1
 c568:  aux_cap_helper__b3,r13,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b3,r13,band_HIGH_PAIRED__Z3 <= 0
 c569:  - X__b0,r7,band_UNPAIRED + aux_cap_helper__b0,r7,band_UNPAIRED__condX2
        + aux_cap_helper__b0,r7,band_UNPAIRED__condX1  = 0
 c570:  - aux_cap_b8,r8,band_LOW_PAIRED
        - 1.4943850124 aux_cap_helper__b8,r8,band_LOW_PAIRED__Z2
        + 3.735962531 aux_cap_helper__b8,r8,band_LOW_PAIRED__condX1
        + 5.2303475434 aux_cap_helper__b8,r8,band_LOW_PAIRED__condX2  = 0
 c571:  - aux_sv_b8,r12 + 46.1022656099 sv_function_helpervar_b8,r12__condX2
        + 27.2033257155 sv_function_helpervar_b8,r12__condX3
        + 1343.0085752554 sv_function_helpervar_b8,r12__condX1
        + 207.435219734485 sv_function_helpervar_b8,r12__Z2
        + 360.482440900956 sv_function_helpervar_b8,r12__Z3  = 0
 c572:  - 6.17377049158038 sv_function_helpervar_b4,r7__Z2
        + sv_function_helpervar_b4,r7__condX2 <= 0
 c573:  - aux_cap_helper__b2,r7,band_LOW_PAIRED__condX1 <= 0
 c574:  0.159946187484381 sv_function_helpervar_b3,r11__Z2
        - sv_function_helpervar_b3,r11__condX2 <= 0
 c575:  - 15.9946187432626 sv_function_helpervar_b8,r13__Z3
        + sv_function_helpervar_b8,r13__condX3 <= 0
 c576:  - sv_function_helpervar_b4,r4__condX3
        + 6.18586368768639 sv_function_helpervar_b4,r4__Z3 <= 0
 c577:  aux_cap_helper__b0,r8,band_LOW_PAIRED__condX1
        - aux_cap_helper__b0,r8,band_LOW_PAIRED__Z1 <= 0
 c578:  - 15.9946187432626 sv_function_helpervar_b0,r9__Z3
        + sv_function_helpervar_b0,r9__condX3 <= 0
 c579:  aux_cap_helper__b7,r13,band_UNPAIRED__condX1
        - aux_cap_helper__b7,r13,band_UNPAIRED__Z1 <= 0
 c580:  sv_function_helpervar_b3,r6__Z1 + sv_function_helpervar_b3,r6__Z3
        + sv_function_helpervar_b3,r6__Z2  = 1
 c581:  - aux_cap_b6,r13,band_LOW_PAIRED
        - 1.4943850124 aux_cap_helper__b6,r13,band_LOW_PAIRED__Z2
        + 3.735962531 aux_cap_helper__b6,r13,band_LOW_PAIRED__condX1
        + 5.2303475434 aux_cap_helper__b6,r13,band_LOW_PAIRED__condX2  = 0
 c582:  - aux_cap_helper__b4,r12,band_HIGH_PAIRED__condX1 <= 0
 c583:  - sv_function_helpervar_b5,r1__condX1 <= 0
 c584:  - 6.40785831161345 sv_function_helpervar_b9,r9__Z2
        + sv_function_helpervar_b9,r9__condX2 <= 0
 c585:  - aux_cap_helper__b7,r11,band_LOW_PAIRED__condX2
        + aux_cap_helper__b7,r11,band_LOW_PAIRED__Z2 <= 0
 c586:  aux_cap_helper__b5,r5,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b5,r5,band_UNPAIRED__Z2 <= 0
 c587:  - sv_function_helpervar_b2,r12__condX1 <= 0
 c588:  sv_function_helpervar_b9,r10__condX1
        - 0.159946187433718 sv_function_helpervar_b9,r10__Z1 <= 0
 c589:  195.899546675363 sv_function_helpervar_b4,r1__Z2
        + 446.358960432527 sv_function_helpervar_b4,r1__Z3 - aux_sv_b4,r1
        + 1289.3272635371 sv_function_helpervar_b4,r1__condX1
        + 64.543166995 sv_function_helpervar_b4,r1__condX2
        + 19.8460407263 sv_function_helpervar_b4,r1__condX3  = 0
 c590:  - aux_c_b8,r0 + sv_function_helpervar_b8,r0__condX3
        + sv_function_helpervar_b8,r0__condX1
        + sv_function_helpervar_b8,r0__condX2  = 0
 c591:  X__b8,r7,band_LOW_PAIRED + X__b8,r7,band_UNPAIRED
        - GlobalBidder_W_ir__b8,r7 + X__b8,r7,band_HIGH_PAIRED >= 0
 c592:  55.9244787157 sv_function_helpervar_b5,r1__condX2
        + 1169.5974422848 sv_function_helpervar_b5,r1__condX1
        + 18.4569463357 sv_function_helpervar_b5,r1__condX3 - aux_sv_b5,r1
        + 397.649853340479 sv_function_helpervar_b5,r1__Z3
        + 178.127744593034 sv_function_helpervar_b5,r1__Z2  = 0
 c593:  aux_cap_helper__b0,r13,band_UNPAIRED__Z2
        - aux_cap_helper__b0,r13,band_UNPAIRED__condX2 <= 0
 c594:  sv_function_helpervar_b9,r0__condX2
        - 7.36804854257395 sv_function_helpervar_b9,r0__Z2 <= 0
 c595:  - sv_function_helpervar_b9,r13__condX3
        + 8.31490900094671 sv_function_helpervar_b9,r13__Z3 <= 0
 c596:  X__b7,r1,band_HIGH_PAIRED + X__b0,r1,band_HIGH_PAIRED
        + X__b6,r1,band_HIGH_PAIRED + X__b3,r1,band_HIGH_PAIRED
        + X__b4,r1,band_HIGH_PAIRED + X__b5,r1,band_HIGH_PAIRED
        + X__b9,r1,band_HIGH_PAIRED + X__b2,r1,band_HIGH_PAIRED
        + X__b1,r1,band_HIGH_PAIRED + X__b8,r1,band_HIGH_PAIRED <= 3
 c597:  - aux_c_b6,r13 + aux_cap_b6,r13,band_LOW_PAIRED
        + aux_cap_b6,r13,band_UNPAIRED + aux_cap_b6,r13,band_HIGH_PAIRED  = 0
 c598:  - aux_Omega_b2,r1 + 221.359496345996 aux_sv_b2,r1  = 0
 c599:  - 2 aux_cap_helper__b4,r11,band_UNPAIRED__Z2
        + aux_cap_helper__b4,r11,band_UNPAIRED__condX2 <= 0
 c600:  - X__b7,r1,band_HIGH_PAIRED
        + aux_cap_helper__b7,r1,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b7,r1,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b7,r1,band_HIGH_PAIRED__condX1  = 0
 c601:  - 18497526.8940961 GlobalBidder_hat{W}__i8,k1 + GlobalBidder_PSI__i8,k1
        <= 0
 c602:  - aux_cap_helper__b0,r11,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b0,r11,band_HIGH_PAIRED__Z3 <= 0
 c603:  2 aux_cap_helper__b3,r12,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b3,r12,band_HIGH_PAIRED__condX3 <= 0
 c604:  sv_function_helpervar_b9,r4__condX2
        - 6.52508471074699 sv_function_helpervar_b9,r4__Z2 <= 0
 c605:  aux_cap_helper__b4,r3,band_UNPAIRED__Z2
        - aux_cap_helper__b4,r3,band_UNPAIRED__condX2 <= 0
 c606:  sv_function_helpervar_b4,r2__Z3 + sv_function_helpervar_b4,r2__Z1
        + sv_function_helpervar_b4,r2__Z2  = 1
 c607:  aux_cap_helper__b9,r13,band_UNPAIRED__condX1
        - aux_cap_helper__b9,r13,band_UNPAIRED__Z1 <= 0
 c608:  aux_cap_helper__b8,r8,band_UNPAIRED__condX1
        + aux_cap_helper__b8,r8,band_UNPAIRED__condX2 - X__b8,r8,band_UNPAIRED
         = 0
 c609:  aux_cap_helper__b3,r13,band_LOW_PAIRED__condX1
        + aux_cap_helper__b3,r13,band_LOW_PAIRED__condX2
        - X__b3,r13,band_LOW_PAIRED  = 0
 c610:  - aux_cap_helper__b1,r5,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b1,r5,band_HIGH_PAIRED__Z2 <= 0
 c611:  - aux_cap_helper__b1,r11,band_LOW_PAIRED__condX2
        + aux_cap_helper__b1,r11,band_LOW_PAIRED__Z2 <= 0
 c612:  - 15.9946187432626 sv_function_helpervar_b3,r1__Z3
        + sv_function_helpervar_b3,r1__condX3 <= 0
 c613:  - aux_cap_helper__b9,r2,band_UNPAIRED__Z1
        + aux_cap_helper__b9,r2,band_UNPAIRED__condX1 <= 0
 c614:  aux_cap_helper__b5,r12,band_UNPAIRED__condX2
        + aux_cap_helper__b5,r12,band_UNPAIRED__condX1
        - X__b5,r12,band_UNPAIRED  = 0
 c615:  sv_function_helpervar_b0,r5__condX1
        - 0.159946187439545 sv_function_helpervar_b0,r5__Z1 <= 0
 c616:  - aux_cap_b2,r3,band_LOW_PAIRED
        - 1.4943850124 aux_cap_helper__b2,r3,band_LOW_PAIRED__Z2
        + 3.735962531 aux_cap_helper__b2,r3,band_LOW_PAIRED__condX1
        + 5.2303475434 aux_cap_helper__b2,r3,band_LOW_PAIRED__condX2  = 0
 c617:  aux_cap_helper__b9,r13,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b9,r13,band_UNPAIRED__Z2 <= 0
 c618:  - aux_cap_helper__b1,r5,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b1,r5,band_HIGH_PAIRED__Z3 <= 0
 c619:  - aux_cap_helper__b7,r13,band_LOW_PAIRED__condX1 <= 0
 c620:  aux_cap_helper__b6,r13,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b6,r13,band_HIGH_PAIRED__Z2 <= 0
 c621:  - sv_function_helpervar_b0,r1__condX1 <= 0
 c622:  - 0.159946187436161 sv_function_helpervar_b3,r1__Z1
        + sv_function_helpervar_b3,r1__condX1 <= 0
 c623:  sv_function_helpervar_b5,r10__Z1 + sv_function_helpervar_b5,r10__Z2
        + sv_function_helpervar_b5,r10__Z3  = 1
 c624:  - aux_cap_b9,r2,band_UNPAIRED
        - 0.3214355447 aux_cap_helper__b9,r2,band_UNPAIRED__Z2
        + 1.1250244065 aux_cap_helper__b9,r2,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b9,r2,band_UNPAIRED__condX1  = 0
 c625:  sv_function_helpervar_b5,r5__condX2
        - 6.26596121586556 sv_function_helpervar_b5,r5__Z2 <= 0
 c626:  aux_cap_helper__b5,r3,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b5,r3,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b5,r3,band_HIGH_PAIRED__Z1  = 1
 c627:  - aux_cap_b8,r11,band_LOW_PAIRED
        + 3.735962531 aux_cap_helper__b8,r11,band_LOW_PAIRED__condX1
        + 5.2303475434 aux_cap_helper__b8,r11,band_LOW_PAIRED__condX2
        - 1.4943850124 aux_cap_helper__b8,r11,band_LOW_PAIRED__Z2  = 0
 c628:  0.0102040816326531 X__b9,r5,band_UNPAIRED
        + 0.0102040816326531 X__b9,r5,band_HIGH_PAIRED
        + 0.0102040816326531 X__b9,r5,band_LOW_PAIRED
        - GlobalBidder_W_ir__b9,r5 <= 0
 c629:  sv_function_helpervar_b6,r11__condX3
        - 15.9946187432626 sv_function_helpervar_b6,r11__Z3 <= 0
 c630:  - aux_cap_b6,r0,band_UNPAIRED
        - 0.3214355447 aux_cap_helper__b6,r0,band_UNPAIRED__Z2
        + 1.1250244065 aux_cap_helper__b6,r0,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b6,r0,band_UNPAIRED__condX1  = 0
 c631:  - aux_cap_helper__b0,r11,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b0,r11,band_HIGH_PAIRED__Z2 <= 0
 c632:  - aux_cap_helper__b8,r1,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b8,r1,band_HIGH_PAIRED__Z3 <= 0
 c633:  - 2 aux_cap_helper__b5,r4,band_LOW_PAIRED__Z2
        + aux_cap_helper__b5,r4,band_LOW_PAIRED__condX2 <= 0
 c634:  aux_cap_helper__b3,r12,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b3,r12,band_HIGH_PAIRED__condX2 <= 0
 c635:  sv_function_helpervar_b8,r7__Z1 + sv_function_helpervar_b8,r7__Z3
        + sv_function_helpervar_b8,r7__Z2  = 1
 c636:  aux_cap_helper__b4,r11,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b4,r11,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b4,r11,band_HIGH_PAIRED__Z1  = 1
 c637:  - aux_Omega_b7,r5 + 721.1346882145 aux_sv_b7,r5  = 0
 c638:  - 2 aux_cap_helper__b6,r3,band_LOW_PAIRED__Z2
        + aux_cap_helper__b6,r3,band_LOW_PAIRED__condX2 <= 0
 c639:  - aux_cap_helper__b4,r1,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b4,r1,band_HIGH_PAIRED__Z2 <= 0
 c640:  - X__b5,r13,band_LOW_PAIRED
        + aux_cap_helper__b5,r13,band_LOW_PAIRED__condX2
        + aux_cap_helper__b5,r13,band_LOW_PAIRED__condX1  = 0
 c641:  - 0.159946187436986 sv_function_helpervar_b4,r1__Z1
        + sv_function_helpervar_b4,r1__condX1 <= 0
 c642:  - aux_c_b7,r3 + aux_cap_b7,r3,band_UNPAIRED
        + aux_cap_b7,r3,band_LOW_PAIRED + aux_cap_b7,r3,band_HIGH_PAIRED  = 0
 c643:  aux_cap_helper__b5,r2,band_UNPAIRED__Z2
        + aux_cap_helper__b5,r2,band_UNPAIRED__Z1  = 1
 c644:  - aux_cap_helper__b7,r5,band_UNPAIRED__condX1 <= 0
 c645:  aux_cap_b5,r6,band_LOW_PAIRED - aux_c_b5,r6
        + aux_cap_b5,r6,band_HIGH_PAIRED + aux_cap_b5,r6,band_UNPAIRED  = 0
 c646:  aux_cap_helper__b3,r6,band_LOW_PAIRED__Z1
        + aux_cap_helper__b3,r6,band_LOW_PAIRED__Z2  = 1
 c647:  - aux_cap_helper__b7,r9,band_LOW_PAIRED__condX1 <= 0
 c648:  - sv_function_helpervar_b5,r10__condX2
        + 0.159946187496824 sv_function_helpervar_b5,r10__Z2 <= 0
 c649:  - X__b0,r10,band_LOW_PAIRED
        + aux_cap_helper__b0,r10,band_LOW_PAIRED__condX2
        + aux_cap_helper__b0,r10,band_LOW_PAIRED__condX1  = 0
 c650:  - 0.159946187436079 sv_function_helpervar_b5,r7__Z1
        + sv_function_helpervar_b5,r7__condX1 <= 0
 c651:  - aux_c_b8,r9 + aux_cap_b8,r9,band_UNPAIRED
        + aux_cap_b8,r9,band_HIGH_PAIRED + aux_cap_b8,r9,band_LOW_PAIRED  = 0
 c652:  aux_cap_helper__b6,r5,band_LOW_PAIRED__Z1
        + aux_cap_helper__b6,r5,band_LOW_PAIRED__Z2  = 1
 c653:  - sv_function_helpervar_b6,r6__condX2
        + 0.159946187491039 sv_function_helpervar_b6,r6__Z2 <= 0
 c654:  aux_cap_helper__b4,r6,band_UNPAIRED__Z2
        + aux_cap_helper__b4,r6,band_UNPAIRED__Z1  = 1
 c655:  X__b7,r1,band_HIGH_PAIRED + X__b7,r1,band_LOW_PAIRED
        - GlobalBidder_W_ir__b7,r1 + X__b7,r1,band_UNPAIRED >= 0
 c656:  - sv_function_helpervar_b6,r2__condX3
        + 5.52779119010926 sv_function_helpervar_b6,r2__Z3 <= 0
 c657:  - 15.9946187432626 sv_function_helpervar_b1,r5__Z3
        + sv_function_helpervar_b1,r5__condX3 <= 0
 c658:  - aux_cap_helper__b6,r7,band_HIGH_PAIRED__condX1 <= 0
 c659:  0.0102040816326531 X__b8,r12,band_UNPAIRED
        + 0.0102040816326531 X__b8,r12,band_HIGH_PAIRED
        - GlobalBidder_W_ir__b8,r12
        + 0.0102040816326531 X__b8,r12,band_LOW_PAIRED <= 0
 c660:  - aux_cap_helper__b0,r2,band_HIGH_PAIRED__condX1 <= 0
 c661:  - aux_sv_b8,r7 + 1343.0085755461 sv_function_helpervar_b8,r7__condX1
        + 27.7249782544 sv_function_helpervar_b8,r7__condX3
        + 45.2552617453 sv_function_helpervar_b8,r7__condX2
        + 352.138807425053 sv_function_helpervar_b8,r7__Z3
        + 207.570694774959 sv_function_helpervar_b8,r7__Z2  = 0
 c662:  - aux_cap_helper__b6,r3,band_LOW_PAIRED__Z1
        + aux_cap_helper__b6,r3,band_LOW_PAIRED__condX1 <= 0
 c663:  - 2 aux_cap_helper__b9,r2,band_UNPAIRED__Z2
        + aux_cap_helper__b9,r2,band_UNPAIRED__condX2 <= 0
 c664:  aux_cap_helper__b0,r2,band_LOW_PAIRED__Z2
        - aux_cap_helper__b0,r2,band_LOW_PAIRED__condX2 <= 0
 c665:  - aux_cap_helper__b0,r12,band_LOW_PAIRED__condX2
        + aux_cap_helper__b0,r12,band_LOW_PAIRED__Z2 <= 0
 c666:  aux_cap_helper__b5,r3,band_UNPAIRED__condX2
        + aux_cap_helper__b5,r3,band_UNPAIRED__condX1 - X__b5,r3,band_UNPAIRED
         = 0
 c667:  aux_cap_helper__b1,r8,band_LOW_PAIRED__Z2
        - aux_cap_helper__b1,r8,band_LOW_PAIRED__condX2 <= 0
 c668:  - aux_c_b2,r6 + sv_function_helpervar_b2,r6__condX2
        + sv_function_helpervar_b2,r6__condX1
        + sv_function_helpervar_b2,r6__condX3  = 0
 c669:  - aux_Omega_b6,r0 + 91.1299728910289 aux_sv_b6,r0  = 0
 c670:  aux_cap_b4,r0,band_HIGH_PAIRED + aux_cap_b4,r0,band_UNPAIRED
        + aux_cap_b4,r0,band_LOW_PAIRED - aux_c_b4,r0  = 0
 c671:  - aux_cap_helper__b4,r2,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b4,r2,band_HIGH_PAIRED__condX1 <= 0
 c672:  - 15.9946187432626 sv_function_helpervar_b0,r10__Z3
        + sv_function_helpervar_b0,r10__condX3 <= 0
 c673:  - X__b0,r9,band_UNPAIRED + aux_cap_helper__b0,r9,band_UNPAIRED__condX1
        + aux_cap_helper__b0,r9,band_UNPAIRED__condX2  = 0
 c674:  - aux_cap_helper__b6,r5,band_UNPAIRED__Z1
        + aux_cap_helper__b6,r5,band_UNPAIRED__condX1 <= 0
 c675:  - aux_cap_b9,r2,band_LOW_PAIRED
        - 1.4943850124 aux_cap_helper__b9,r2,band_LOW_PAIRED__Z2
        + 5.2303475434 aux_cap_helper__b9,r2,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b9,r2,band_LOW_PAIRED__condX1  = 0
 c676:  - aux_cap_helper__b9,r3,band_LOW_PAIRED__condX2
        + aux_cap_helper__b9,r3,band_LOW_PAIRED__Z2 <= 0
 c677:  - 3 aux_cap_helper__b8,r8,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b8,r8,band_HIGH_PAIRED__condX3 <= 0
 c678:  aux_cap_helper__b3,r11,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b3,r11,band_HIGH_PAIRED__Z2 <= 0
 c679:  - 2 aux_cap_helper__b8,r10,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b8,r10,band_HIGH_PAIRED__condX2 <= 0
 c680:  2.1635071398 aux_cap_helper__b6,r13,band_HIGH_PAIRED__condX2
        + 1.3908260184 aux_cap_helper__b6,r13,band_HIGH_PAIRED__condX3
        + 1.5453622427 aux_cap_helper__b6,r13,band_HIGH_PAIRED__condX1
        - 0.6181448971 aux_cap_helper__b6,r13,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b6,r13,band_HIGH_PAIRED__Z3
        - aux_cap_b6,r13,band_HIGH_PAIRED  = 0
 c681:  - 3 aux_cap_helper__b7,r5,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b7,r5,band_HIGH_PAIRED__condX3 <= 0
 c682:  aux_cap_helper__b9,r7,band_UNPAIRED__Z1
        + aux_cap_helper__b9,r7,band_UNPAIRED__Z2  = 1
 c683:  - aux_cap_helper__b1,r6,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b1,r6,band_HIGH_PAIRED__Z3 <= 0
 c684:  - aux_sv_b5,r2 + 389.716488508378 sv_function_helpervar_b5,r2__Z3
        + 178.5255525999 sv_function_helpervar_b5,r2__Z2
        + 53.437342167 sv_function_helpervar_b5,r2__condX2
        + 18.9529484574 sv_function_helpervar_b5,r2__condX3
        + 1169.5974420414 sv_function_helpervar_b5,r2__condX1  = 0
 c685:  - X__b2,r10,band_LOW_PAIRED
        + aux_cap_helper__b2,r10,band_LOW_PAIRED__condX2
        + aux_cap_helper__b2,r10,band_LOW_PAIRED__condX1  = 0
 c686:  - 2 aux_cap_helper__b6,r5,band_UNPAIRED__Z2
        + aux_cap_helper__b6,r5,band_UNPAIRED__condX2 <= 0
 c687:  aux_cap_helper__b1,r6,band_UNPAIRED__condX1
        - aux_cap_helper__b1,r6,band_UNPAIRED__Z1 <= 0
 c688:  aux_cap_helper__b9,r10,band_LOW_PAIRED__Z2
        + aux_cap_helper__b9,r10,band_LOW_PAIRED__Z1  = 1
 c689:  - 2 aux_cap_helper__b4,r2,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b4,r2,band_HIGH_PAIRED__condX2 <= 0
 c690:  sv_function_helpervar_b3,r11__condX1
        + sv_function_helpervar_b3,r11__condX2
        + sv_function_helpervar_b3,r11__condX3 - aux_c_b3,r11  = 0
 c691:  3.735962531 aux_cap_helper__b5,r7,band_LOW_PAIRED__condX1
        + 5.2303475434 aux_cap_helper__b5,r7,band_LOW_PAIRED__condX2
        - 1.4943850124 aux_cap_helper__b5,r7,band_LOW_PAIRED__Z2
        - aux_cap_b5,r7,band_LOW_PAIRED  = 0
 c692:  - sv_function_helpervar_b7,r7__condX1 <= 0
 c693:  - aux_cap_helper__b7,r10,band_UNPAIRED__condX2
        + aux_cap_helper__b7,r10,band_UNPAIRED__Z2 <= 0
 c694:  sv_function_helpervar_b9,r0__condX1
        - 0.159946187463129 sv_function_helpervar_b9,r0__Z1 <= 0
 c695:  - aux_Omega_b5,r1 + 212.734270284368 aux_sv_b5,r1  = 0
 c696:  - 15.9946187432626 sv_function_helpervar_b1,r8__Z3
        + sv_function_helpervar_b1,r8__condX3 <= 0
 c697:  - aux_Omega_b3,r2 + 172.934954719563 aux_sv_b3,r2  = 0
 c698:  473.042659538745 sv_function_helpervar_b3,r1__Z3
        + 209.886488101416 sv_function_helpervar_b3,r1__Z2
        + 21.5207594829 sv_function_helpervar_b3,r1__condX3
        + 1379.5885617582 sv_function_helpervar_b3,r1__condX1
        + 67.3566701091 sv_function_helpervar_b3,r1__condX2 - aux_sv_b3,r1  = 0
 c699:  aux_cap_helper__b5,r3,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b5,r3,band_UNPAIRED__Z2 <= 0
 c700:  aux_cap_helper__b1,r6,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b1,r6,band_UNPAIRED__Z2 <= 0
 c701:  - sv_function_helpervar_b9,r3__condX1 <= 0
 c702:  2 aux_cap_helper__b5,r4,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b5,r4,band_HIGH_PAIRED__condX3 <= 0
 c703:  - aux_c_b7,r11 + aux_cap_b7,r11,band_HIGH_PAIRED
        + aux_cap_b7,r11,band_LOW_PAIRED + aux_cap_b7,r11,band_UNPAIRED  = 0
 c704:  2 aux_cap_helper__b0,r12,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b0,r12,band_HIGH_PAIRED__condX3 <= 0
 c705:  aux_cap_helper__b7,r9,band_UNPAIRED__condX2
        + aux_cap_helper__b7,r9,band_UNPAIRED__condX1 - X__b7,r9,band_UNPAIRED
         = 0
 c706:  - sv_function_helpervar_b8,r12__condX3
        + 8.09819079914792 sv_function_helpervar_b8,r12__Z3 <= 0
 c707:  - aux_cap_helper__b2,r0,band_LOW_PAIRED__condX1 <= 0
 c708:  sv_function_helpervar_b6,r9__condX3
        - 15.9946187432626 sv_function_helpervar_b6,r9__Z3 <= 0
 c709:  - aux_cap_helper__b8,r10,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b8,r10,band_HIGH_PAIRED__condX1 <= 0
 c710:  aux_cap_helper__b0,r4,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b0,r4,band_UNPAIRED__Z2 <= 0
 c711:  6.5782911246891 sv_function_helpervar_b2,r13__Z3
        - sv_function_helpervar_b2,r13__condX3 <= 0
 c712:  sv_function_helpervar_b0,r1__Z1 + sv_function_helpervar_b0,r1__Z3
        + sv_function_helpervar_b0,r1__Z2  = 1
 c713:  aux_cap_helper__b3,r11,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b3,r11,band_HIGH_PAIRED__Z1 <= 0
 c714:  aux_cap_helper__b4,r13,band_LOW_PAIRED__Z2
        + aux_cap_helper__b4,r13,band_LOW_PAIRED__Z1  = 1
 c715:  - 6.09869560127261 sv_function_helpervar_b1,r0__Z2
        + sv_function_helpervar_b1,r0__condX2 <= 0
 c716:  aux_cap_helper__b2,r13,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b2,r13,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b2,r13,band_HIGH_PAIRED__Z1  = 1
 c717:  - aux_Omega_b9,r2 + 281.909077772859 aux_sv_b9,r2  = 0
 c718:  aux_cap_helper__b2,r9,band_UNPAIRED__Z1
        + aux_cap_helper__b2,r9,band_UNPAIRED__Z2  = 1
 c719:  aux_cap_b7,r2,band_LOW_PAIRED - aux_c_b7,r2
        + aux_cap_b7,r2,band_HIGH_PAIRED + aux_cap_b7,r2,band_UNPAIRED  = 0
 c720:  - X__b0,r11,band_HIGH_PAIRED
        + aux_cap_helper__b0,r11,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b0,r11,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b0,r11,band_HIGH_PAIRED__condX1  = 0
 c721:  - aux_cap_helper__b5,r8,band_UNPAIRED__condX1 <= 0
 c722:  aux_cap_helper__b5,r3,band_UNPAIRED__condX1
        - aux_cap_helper__b5,r3,band_UNPAIRED__Z1 <= 0
 c723:  aux_cap_helper__b2,r6,band_UNPAIRED__condX2
        + aux_cap_helper__b2,r6,band_UNPAIRED__condX1 - X__b2,r6,band_UNPAIRED
         = 0
 c724:  - aux_cap_helper__b0,r9,band_LOW_PAIRED__condX2
        + aux_cap_helper__b0,r9,band_LOW_PAIRED__Z2 <= 0
 c725:  - 6.63228196246224 sv_function_helpervar_b2,r6__Z2
        + sv_function_helpervar_b2,r6__condX2 <= 0
 c726:  - 15.9946187432626 sv_function_helpervar_b0,r7__Z3
        + sv_function_helpervar_b0,r7__condX3 <= 0
 c727:  1.1250244065 aux_cap_helper__b4,r6,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b4,r6,band_UNPAIRED__condX1
        - aux_cap_b4,r6,band_UNPAIRED
        - 0.3214355447 aux_cap_helper__b4,r6,band_UNPAIRED__Z2  = 0
 c728:  aux_cap_helper__b5,r4,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b5,r4,band_HIGH_PAIRED__condX2 <= 0
 c729:  aux_cap_helper__b0,r3,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b0,r3,band_HIGH_PAIRED__Z1 <= 0
 c730:  - X__b2,r8,band_LOW_PAIRED
        + aux_cap_helper__b2,r8,band_LOW_PAIRED__condX1
        + aux_cap_helper__b2,r8,band_LOW_PAIRED__condX2  = 0
 c731:  aux_cap_helper__b6,r13,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b6,r13,band_HIGH_PAIRED__Z1 <= 0
 c732:  sv_function_helpervar_b9,r13__Z2 + sv_function_helpervar_b9,r13__Z3
        + sv_function_helpervar_b9,r13__Z1  = 1
 c733:  aux_cap_helper__b1,r1,band_UNPAIRED__Z2
        - aux_cap_helper__b1,r1,band_UNPAIRED__condX2 <= 0
 c734:  aux_cap_helper__b8,r2,band_UNPAIRED__Z2
        - aux_cap_helper__b8,r2,band_UNPAIRED__condX2 <= 0
 c735:  aux_Omega_b2,r6 + aux_Omega_b2,r1 + aux_Omega_b2,r13 - v__2  = 0
 c736:  - aux_cap_helper__b8,r1,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b8,r1,band_HIGH_PAIRED__Z2 <= 0
 c737:  - aux_cap_helper__b1,r12,band_UNPAIRED__condX1 <= 0
 c738:  - X__b6,r2,band_UNPAIRED + aux_cap_helper__b6,r2,band_UNPAIRED__condX1
        + aux_cap_helper__b6,r2,band_UNPAIRED__condX2  = 0
 c739:  sv_function_helpervar_b6,r1__condX1
        - 0.159946187477713 sv_function_helpervar_b6,r1__Z1 <= 0
 c740:  aux_cap_helper__b0,r1,band_UNPAIRED__Z1
        + aux_cap_helper__b0,r1,band_UNPAIRED__Z2  = 1
 c741:  - X__b7,r0,band_UNPAIRED + aux_cap_helper__b7,r0,band_UNPAIRED__condX2
        + aux_cap_helper__b7,r0,band_UNPAIRED__condX1  = 0
 c742:  aux_cap_helper__b0,r3,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b0,r3,band_HIGH_PAIRED__Z2 <= 0
 c743:  2.1635071398 aux_cap_helper__b1,r13,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b1,r13,band_HIGH_PAIRED__condX1
        + 1.3908260184 aux_cap_helper__b1,r13,band_HIGH_PAIRED__condX3
        - aux_cap_b1,r13,band_HIGH_PAIRED
        - 0.6181448971 aux_cap_helper__b1,r13,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b1,r13,band_HIGH_PAIRED__Z3  = 0
 c744:  - X__b3,r2,band_LOW_PAIRED
        + aux_cap_helper__b3,r2,band_LOW_PAIRED__condX2
        + aux_cap_helper__b3,r2,band_LOW_PAIRED__condX1  = 0
 c745:  aux_cap_helper__b7,r0,band_UNPAIRED__Z2
        - aux_cap_helper__b7,r0,band_UNPAIRED__condX2 <= 0
 c746:  aux_cap_helper__b0,r0,band_UNPAIRED__condX2
        + aux_cap_helper__b0,r0,band_UNPAIRED__condX1 - X__b0,r0,band_UNPAIRED
         = 0
 c747:  - aux_cap_helper__b2,r4,band_LOW_PAIRED__Z1
        + aux_cap_helper__b2,r4,band_LOW_PAIRED__condX1 <= 0
 c748:  - aux_cap_helper__b7,r10,band_LOW_PAIRED__condX1 <= 0
 c749:  - 15.9946187432626 sv_function_helpervar_b2,r6__Z3
        + sv_function_helpervar_b2,r6__condX3 <= 0
 c750:  0.159946187465026 sv_function_helpervar_b4,r7__Z2
        - sv_function_helpervar_b4,r7__condX2 <= 0
 c751:  - aux_cap_helper__b8,r8,band_LOW_PAIRED__condX1 <= 0
 c752:  - aux_cap_helper__b4,r11,band_UNPAIRED__Z1
        + aux_cap_helper__b4,r11,band_UNPAIRED__condX1 <= 0
 c753:  - X__b9,r12,band_LOW_PAIRED
        + aux_cap_helper__b9,r12,band_LOW_PAIRED__condX2
        + aux_cap_helper__b9,r12,band_LOW_PAIRED__condX1  = 0
 c754:  - aux_Omega_b2,r0 + 135.161063228824 aux_sv_b2,r0  = 0
 c755:  aux_cap_helper__b5,r6,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b5,r6,band_HIGH_PAIRED__condX2 <= 0
 c756:  0.0102040816326531 X__b7,r7,band_UNPAIRED
        + 0.0102040816326531 X__b7,r7,band_LOW_PAIRED
        - GlobalBidder_W_ir__b7,r7
        + 0.0102040816326531 X__b7,r7,band_HIGH_PAIRED <= 0
 c757:  aux_Omega_b9,r1 + aux_Omega_b9,r0 + aux_Omega_b9,r3 + aux_Omega_b9,r2
        + aux_Omega_b9,r5 + aux_Omega_b9,r4 + aux_Omega_b9,r7 + aux_Omega_b9,r6
        + aux_Omega_b9,r9 + aux_Omega_b9,r8 - GlobalBidder_PSI__i9,k1
        + aux_Omega_b9,r13 + aux_Omega_b9,r10 + aux_Omega_b9,r11
        + aux_Omega_b9,r12 - 18497526.8940961 GlobalBidder_hat{W}__i9,k1
        >= -18497526.8940961
 c758:  586.031531484115 sv_function_helpervar_b7,r10__Z3
        + 283.648276377593 sv_function_helpervar_b7,r10__Z2 - aux_sv_b7,r10
        + 31.8911999971 sv_function_helpervar_b7,r10__condX3
        + 1850.3233259602 sv_function_helpervar_b7,r10__condX1
        + 76.9251541993 sv_function_helpervar_b7,r10__condX2  = 0
 c759:  - 2 aux_cap_helper__b2,r4,band_LOW_PAIRED__Z2
        + aux_cap_helper__b2,r4,band_LOW_PAIRED__condX2 <= 0
 c760:  aux_cap_helper__b5,r7,band_LOW_PAIRED__condX1
        - aux_cap_helper__b5,r7,band_LOW_PAIRED__Z1 <= 0
 c761:  - aux_c_b5,r4 + sv_function_helpervar_b5,r4__condX3
        + sv_function_helpervar_b5,r4__condX2
        + sv_function_helpervar_b5,r4__condX1  = 0
 c762:  0.15994618743616 sv_function_helpervar_b9,r9__Z2
        - sv_function_helpervar_b9,r9__condX2 <= 0
 c763:  sv_function_helpervar_b2,r3__condX3
        + sv_function_helpervar_b2,r3__condX2
        + sv_function_helpervar_b2,r3__condX1 - aux_c_b2,r3  = 0
 c764:  sv_function_helpervar_b6,r13__condX2
        - 5.52627638616526 sv_function_helpervar_b6,r13__Z2 <= 0
 c765:  2 aux_cap_helper__b8,r11,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b8,r11,band_HIGH_PAIRED__condX3 <= 0
 c766:  - aux_cap_helper__b9,r12,band_HIGH_PAIRED__condX1 <= 0
 c767:  - X__b9,r3,band_UNPAIRED + aux_cap_helper__b9,r3,band_UNPAIRED__condX2
        + aux_cap_helper__b9,r3,band_UNPAIRED__condX1  = 0
 c768:  aux_cap_helper__b8,r7,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b8,r7,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b8,r7,band_HIGH_PAIRED__condX2
        - X__b8,r7,band_HIGH_PAIRED  = 0
 c769:  - aux_cap_helper__b9,r5,band_HIGH_PAIRED__condX1 <= 0
 c770:  - aux_cap_helper__b9,r11,band_LOW_PAIRED__condX1 <= 0
 c771:  2 aux_cap_helper__b5,r6,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b5,r6,band_HIGH_PAIRED__condX3 <= 0
 c772:  - sv_function_helpervar_b1,r11__condX3
        + 5.94510944286213 sv_function_helpervar_b1,r11__Z3 <= 0
 c773:  aux_cap_helper__b6,r11,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b6,r11,band_HIGH_PAIRED__Z1 <= 0
 c774:  - sv_function_helpervar_b0,r3__condX3
        + 6.67198907969296 sv_function_helpervar_b0,r3__Z3 <= 0
 c775:  5.2303475434 aux_cap_helper__b0,r6,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b0,r6,band_LOW_PAIRED__condX1
        - 1.4943850124 aux_cap_helper__b0,r6,band_LOW_PAIRED__Z2
        - aux_cap_b0,r6,band_LOW_PAIRED  = 0
 c776:  - X__b9,r0,band_HIGH_PAIRED
        + aux_cap_helper__b9,r0,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b9,r0,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b9,r0,band_HIGH_PAIRED__condX2  = 0
 c777:  aux_cap_helper__b9,r13,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b9,r13,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b9,r13,band_HIGH_PAIRED__Z2  = 1
 c778:  - aux_cap_helper__b0,r0,band_HIGH_PAIRED__condX1 <= 0
 c779:  - aux_Omega_b3,r1 + 189.120450868004 aux_sv_b3,r1  = 0
 c780:  aux_cap_helper__b2,r13,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b2,r13,band_UNPAIRED__Z2 <= 0
 c781:  aux_cap_b3,r8,band_HIGH_PAIRED + aux_cap_b3,r8,band_LOW_PAIRED
        - aux_c_b3,r8 + aux_cap_b3,r8,band_UNPAIRED  = 0
 c782:  aux_cap_helper__b2,r12,band_UNPAIRED__Z2
        + aux_cap_helper__b2,r12,band_UNPAIRED__Z1  = 1
 c783:  sv_function_helpervar_b8,r2__condX3
        - 15.9946187432626 sv_function_helpervar_b8,r2__Z3 <= 0
 c784:  - X__b5,r9,band_HIGH_PAIRED
        + aux_cap_helper__b5,r9,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b5,r9,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b5,r9,band_HIGH_PAIRED__condX1  = 0
 c785:  - aux_cap_helper__b5,r1,band_LOW_PAIRED__Z1
        + aux_cap_helper__b5,r1,band_LOW_PAIRED__condX1 <= 0
 c786:  aux_cap_helper__b6,r4,band_LOW_PAIRED__condX1
        + aux_cap_helper__b6,r4,band_LOW_PAIRED__condX2
        - X__b6,r4,band_LOW_PAIRED  = 0
 c787:  - sv_function_helpervar_b6,r0__condX1 <= 0
 c788:  - aux_cap_helper__b9,r2,band_HIGH_PAIRED__condX1 <= 0
 c789:  - aux_cap_helper__b0,r6,band_LOW_PAIRED__condX2
        + aux_cap_helper__b0,r6,band_LOW_PAIRED__Z2 <= 0
 c790:  aux_cap_helper__b8,r11,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b8,r11,band_HIGH_PAIRED__condX2 <= 0
 c791:  - aux_cap_b7,r4,band_UNPAIRED
        + 0.8035888618 aux_cap_helper__b7,r4,band_UNPAIRED__condX1
        + 1.1250244065 aux_cap_helper__b7,r4,band_UNPAIRED__condX2
        - 0.3214355447 aux_cap_helper__b7,r4,band_UNPAIRED__Z2  = 0
 c792:  0.159946187438714 sv_function_helpervar_b0,r10__Z2
        - sv_function_helpervar_b0,r10__condX2 <= 0
 c793:  aux_cap_helper__b3,r10,band_UNPAIRED__Z2
        + aux_cap_helper__b3,r10,band_UNPAIRED__Z1  = 1
 c794:  sv_function_helpervar_b0,r3__condX1
        - 0.159946187448897 sv_function_helpervar_b0,r3__Z1 <= 0
 c795:  - 2 aux_cap_helper__b5,r1,band_LOW_PAIRED__Z2
        + aux_cap_helper__b5,r1,band_LOW_PAIRED__condX2 <= 0
 c796:  - aux_c_b5,r9 + aux_cap_b5,r9,band_UNPAIRED
        + aux_cap_b5,r9,band_HIGH_PAIRED + aux_cap_b5,r9,band_LOW_PAIRED  = 0
 c797:  aux_cap_helper__b0,r13,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b0,r13,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b0,r13,band_HIGH_PAIRED__condX3
        - X__b0,r13,band_HIGH_PAIRED  = 0
 c798:  aux_cap_helper__b9,r0,band_LOW_PAIRED__condX1
        + aux_cap_helper__b9,r0,band_LOW_PAIRED__condX2
        - X__b9,r0,band_LOW_PAIRED  = 0
 c799:  aux_cap_helper__b6,r0,band_LOW_PAIRED__condX1
        - aux_cap_helper__b6,r0,band_LOW_PAIRED__Z1 <= 0
 c800:  - X__b9,r11,band_LOW_PAIRED
        + aux_cap_helper__b9,r11,band_LOW_PAIRED__condX1
        + aux_cap_helper__b9,r11,band_LOW_PAIRED__condX2  = 0
 c801:  0.8035888618 aux_cap_helper__b4,r2,band_UNPAIRED__condX1
        + 1.1250244065 aux_cap_helper__b4,r2,band_UNPAIRED__condX2
        - aux_cap_b4,r2,band_UNPAIRED
        - 0.3214355447 aux_cap_helper__b4,r2,band_UNPAIRED__Z2  = 0
 c802:  sv_function_helpervar_b9,r11__condX1
        - 0.159946187514989 sv_function_helpervar_b9,r11__Z1 <= 0
 c803:  aux_cap_helper__b0,r4,band_UNPAIRED__condX1
        - aux_cap_helper__b0,r4,band_UNPAIRED__Z1 <= 0
 c804:  - aux_cap_helper__b2,r11,band_UNPAIRED__condX2
        + aux_cap_helper__b2,r11,band_UNPAIRED__Z2 <= 0
 c805:  aux_cap_b1,r12,band_HIGH_PAIRED + aux_cap_b1,r12,band_UNPAIRED
        + aux_cap_b1,r12,band_LOW_PAIRED - aux_c_b1,r12  = 0
 c806:  - sv_function_helpervar_b7,r11__condX2
        + 0.159946187437288 sv_function_helpervar_b7,r11__Z2 <= 0
 c807:  sv_function_helpervar_b9,r2__condX2
        + sv_function_helpervar_b9,r2__condX3
        + sv_function_helpervar_b9,r2__condX1 - aux_c_b9,r2  = 0
 c808:  aux_cap_helper__b6,r11,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b6,r11,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b6,r11,band_HIGH_PAIRED__condX1
        - X__b6,r11,band_HIGH_PAIRED  = 0
 c809:  - aux_cap_helper__b6,r11,band_LOW_PAIRED__condX1 <= 0
 c810:  aux_cap_helper__b6,r11,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b6,r11,band_HIGH_PAIRED__Z2 <= 0
 c811:  aux_cap_b1,r0,band_UNPAIRED - aux_c_b1,r0
        + aux_cap_b1,r0,band_LOW_PAIRED + aux_cap_b1,r0,band_HIGH_PAIRED  = 0
 c812:  aux_cap_helper__b3,r2,band_LOW_PAIRED__Z2
        - aux_cap_helper__b3,r2,band_LOW_PAIRED__condX2 <= 0
 c813:  0.159946187432665 sv_function_helpervar_b0,r9__Z2
        - sv_function_helpervar_b0,r9__condX2 <= 0
 c814:  - sv_function_helpervar_b3,r3__condX1 <= 0
 c815:  - X__b5,r0,band_HIGH_PAIRED
        + aux_cap_helper__b5,r0,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b5,r0,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b5,r0,band_HIGH_PAIRED__condX3  = 0
 c816:  sv_function_helpervar_b4,r13__Z1 + sv_function_helpervar_b4,r13__Z2
        + sv_function_helpervar_b4,r13__Z3  = 1
 c817:  - X__b2,r7,band_LOW_PAIRED
        + aux_cap_helper__b2,r7,band_LOW_PAIRED__condX2
        + aux_cap_helper__b2,r7,band_LOW_PAIRED__condX1  = 0
 c818:  - 1.4943850124 aux_cap_helper__b4,r10,band_LOW_PAIRED__Z2
        - aux_cap_b4,r10,band_LOW_PAIRED
        + 5.2303475434 aux_cap_helper__b4,r10,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b4,r10,band_LOW_PAIRED__condX1  = 0
 c819:  - 3 aux_cap_helper__b4,r7,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b4,r7,band_HIGH_PAIRED__condX3 <= 0
 c820:  3.735962531 aux_cap_helper__b3,r13,band_LOW_PAIRED__condX1
        + 5.2303475434 aux_cap_helper__b3,r13,band_LOW_PAIRED__condX2
        - aux_cap_b3,r13,band_LOW_PAIRED
        - 1.4943850124 aux_cap_helper__b3,r13,band_LOW_PAIRED__Z2  = 0
 c821:  21.9311429048 sv_function_helpervar_b3,r8__condX3
        + 65.1181572509 sv_function_helpervar_b3,r8__condX2
        + 1379.5885614955 sv_function_helpervar_b3,r8__condX1
        + 466.478733166431 sv_function_helpervar_b3,r8__Z3
        + 210.244529696627 sv_function_helpervar_b3,r8__Z2 - aux_sv_b3,r8  = 0
 c822:  - aux_cap_helper__b8,r12,band_LOW_PAIRED__condX1 <= 0
 c823:  - 6.59858228882779 sv_function_helpervar_b0,r11__Z2
        + sv_function_helpervar_b0,r11__condX2 <= 0
 c824:  - aux_cap_helper__b3,r10,band_UNPAIRED__condX1 <= 0
 c825:  - aux_cap_helper__b2,r9,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b2,r9,band_HIGH_PAIRED__Z3 <= 0
 c826:  aux_cap_helper__b8,r1,band_LOW_PAIRED__condX2
        + aux_cap_helper__b8,r1,band_LOW_PAIRED__condX1
        - X__b8,r1,band_LOW_PAIRED  = 0
 c827:  sv_function_helpervar_b2,r5__Z2 + sv_function_helpervar_b2,r5__Z3
        + sv_function_helpervar_b2,r5__Z1  = 1
 c828:  aux_cap_helper__b9,r0,band_LOW_PAIRED__Z2
        - aux_cap_helper__b9,r0,band_LOW_PAIRED__condX2 <= 0
 c829:  - aux_cap_helper__b1,r4,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b1,r4,band_HIGH_PAIRED__condX1 <= 0
 c830:  aux_cap_helper__b6,r0,band_UNPAIRED__Z1
        + aux_cap_helper__b6,r0,band_UNPAIRED__Z2  = 1
 c831:  aux_cap_helper__b3,r5,band_UNPAIRED__Z1
        + aux_cap_helper__b3,r5,band_UNPAIRED__Z2  = 1
 c832:  X__b8,r10,band_LOW_PAIRED + X__b1,r10,band_LOW_PAIRED
        + X__b9,r10,band_LOW_PAIRED + X__b2,r10,band_LOW_PAIRED
        + X__b0,r10,band_LOW_PAIRED + X__b3,r10,band_LOW_PAIRED
        + X__b4,r10,band_LOW_PAIRED + X__b7,r10,band_LOW_PAIRED
        + X__b5,r10,band_LOW_PAIRED + X__b6,r10,band_LOW_PAIRED <= 2
 c833:  aux_cap_helper__b2,r2,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b2,r2,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b2,r2,band_HIGH_PAIRED__condX1
        - X__b2,r2,band_HIGH_PAIRED  = 0
 c834:  - sv_function_helpervar_b4,r12__condX1 <= 0
 c835:  aux_cap_helper__b4,r10,band_LOW_PAIRED__Z2
        + aux_cap_helper__b4,r10,band_LOW_PAIRED__Z1  = 1
 c836:  6.33877630346304 sv_function_helpervar_b7,r8__Z3
        - sv_function_helpervar_b7,r8__condX3 <= 0
 c837:  216.44980085505 aux_sv_b0,r9 - aux_Omega_b0,r9  = 0
 c838:  - 2 aux_cap_helper__b1,r4,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b1,r4,band_HIGH_PAIRED__condX2 <= 0
 c839:  sv_function_helpervar_b4,r2__condX3
        - 15.9946187432626 sv_function_helpervar_b4,r2__Z3 <= 0
 c840:  aux_cap_helper__b7,r4,band_LOW_PAIRED__Z1
        + aux_cap_helper__b7,r4,band_LOW_PAIRED__Z2  = 1
 c841:  - aux_cap_helper__b6,r5,band_HIGH_PAIRED__condX1 <= 0
 c842:  - sv_function_helpervar_b7,r7__condX3
        + 7.00008283627391 sv_function_helpervar_b7,r7__Z3 <= 0
 c843:  - sv_function_helpervar_b6,r9__condX1 <= 0
 c844:  aux_cap_helper__b7,r11,band_LOW_PAIRED__condX2
        + aux_cap_helper__b7,r11,band_LOW_PAIRED__condX1
        - X__b7,r11,band_LOW_PAIRED  = 0
 c845:  sv_function_helpervar_b6,r12__Z3 + sv_function_helpervar_b6,r12__Z2
        + sv_function_helpervar_b6,r12__Z1  = 1
 c846:  0.8035888618 aux_cap_helper__b9,r6,band_UNPAIRED__condX1
        + 1.1250244065 aux_cap_helper__b9,r6,band_UNPAIRED__condX2
        - 0.3214355447 aux_cap_helper__b9,r6,band_UNPAIRED__Z2
        - aux_cap_b9,r6,band_UNPAIRED  = 0
 c847:  sv_function_helpervar_b9,r3__condX3
        - 15.9946187432626 sv_function_helpervar_b9,r3__Z3 <= 0
 c848:  - X__b2,r11,band_LOW_PAIRED
        + aux_cap_helper__b2,r11,band_LOW_PAIRED__condX2
        + aux_cap_helper__b2,r11,band_LOW_PAIRED__condX1  = 0
 c849:  sv_function_helpervar_b2,r2__condX3
        + sv_function_helpervar_b2,r2__condX2
        + sv_function_helpervar_b2,r2__condX1 - aux_c_b2,r2  = 0
 c850:  - aux_cap_helper__b1,r7,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b1,r7,band_HIGH_PAIRED__condX1 <= 0
 c851:  sv_function_helpervar_b2,r4__condX2
        - 6.57835953923278 sv_function_helpervar_b2,r4__Z2 <= 0
 c852:  sv_function_helpervar_b7,r7__condX2
        + sv_function_helpervar_b7,r7__condX1
        + sv_function_helpervar_b7,r7__condX3 - aux_c_b7,r7  = 0
 c853:  - 3 aux_cap_helper__b8,r6,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b8,r6,band_HIGH_PAIRED__condX3 <= 0
 c854:  - sv_function_helpervar_b7,r12__condX1 <= 0
 c855:  sv_function_helpervar_b4,r6__Z3 + sv_function_helpervar_b4,r6__Z1
        + sv_function_helpervar_b4,r6__Z2  = 1
 c856:  - aux_cap_helper__b5,r12,band_LOW_PAIRED__condX1 <= 0
 c857:  - aux_cap_helper__b8,r11,band_UNPAIRED__condX1 <= 0
 c858:  aux_cap_helper__b1,r9,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b1,r9,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b1,r9,band_HIGH_PAIRED__condX1
        - X__b1,r9,band_HIGH_PAIRED  = 0
 c859:  - aux_cap_b6,r11,band_HIGH_PAIRED
        + 1.3908260184 aux_cap_helper__b6,r11,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b6,r11,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b6,r11,band_HIGH_PAIRED__condX1
        - 0.6181448971 aux_cap_helper__b6,r11,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b6,r11,band_HIGH_PAIRED__Z3  = 0
 c860:  aux_cap_helper__b7,r7,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b7,r7,band_HIGH_PAIRED__Z3 <= 0
 c861:  - sv_function_helpervar_b1,r4__condX2
        + 0.159946187439907 sv_function_helpervar_b1,r4__Z2 <= 0
 c862:  - 2 aux_cap_helper__b1,r7,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b1,r7,band_HIGH_PAIRED__condX2 <= 0
 c863:  - aux_c_b7,r8 + sv_function_helpervar_b7,r8__condX2
        + sv_function_helpervar_b7,r8__condX3
        + sv_function_helpervar_b7,r8__condX1  = 0
 c864:  - 2 aux_cap_helper__b4,r4,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b4,r4,band_HIGH_PAIRED__condX2 <= 0
 c865:  0.927217345653617 aux_cap_helper__b7,r0,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b7,r0,band_HIGH_PAIRED__Z2
        - aux_cap_b7,r0,band_HIGH_PAIRED
        + 1.3908260184 aux_cap_helper__b7,r0,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b7,r0,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b7,r0,band_HIGH_PAIRED__condX1  = 0
 c866:  - aux_cap_helper__b4,r1,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b4,r1,band_HIGH_PAIRED__Z3 <= 0
 c867:  - sv_function_helpervar_b9,r5__condX2
        + 0.159946187454291 sv_function_helpervar_b9,r5__Z2 <= 0
 c868:  - aux_cap_helper__b2,r6,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b2,r6,band_HIGH_PAIRED__Z2 <= 0
 c869:  aux_cap_helper__b2,r13,band_UNPAIRED__condX1
        - aux_cap_helper__b2,r13,band_UNPAIRED__Z1 <= 0
 c870:  - 0.3214355447 aux_cap_helper__b1,r4,band_UNPAIRED__Z2
        - aux_cap_b1,r4,band_UNPAIRED
        + 0.8035888618 aux_cap_helper__b1,r4,band_UNPAIRED__condX1
        + 1.1250244065 aux_cap_helper__b1,r4,band_UNPAIRED__condX2  = 0
 c871:  aux_cap_helper__b4,r2,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b4,r2,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b4,r2,band_HIGH_PAIRED__Z2  = 1
 c872:  - aux_cap_helper__b2,r6,band_UNPAIRED__condX1 <= 0
 c873:  - aux_cap_helper__b2,r6,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b2,r6,band_HIGH_PAIRED__Z3 <= 0
 c874:  sv_function_helpervar_b3,r7__condX1
        - 0.159946187433093 sv_function_helpervar_b3,r7__Z1 <= 0
 c875:  - sv_function_helpervar_b4,r4__condX1 <= 0
 c876:  - sv_function_helpervar_b7,r2__condX2
        + 0.159946187532419 sv_function_helpervar_b7,r2__Z2 <= 0
 c877:  aux_cap_helper__b3,r2,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b3,r2,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b3,r2,band_HIGH_PAIRED__Z3  = 1
 c878:  6.32530579601375 sv_function_helpervar_b4,r3__Z3
        - sv_function_helpervar_b4,r3__condX3 <= 0
 c879:  - aux_c_b0,r8 + sv_function_helpervar_b0,r8__condX1
        + sv_function_helpervar_b0,r8__condX2
        + sv_function_helpervar_b0,r8__condX3  = 0
 c880:  sv_function_helpervar_b2,r10__condX1
        - 0.159946187444046 sv_function_helpervar_b2,r10__Z1 <= 0
 c881:  - X__b3,r2,band_HIGH_PAIRED
        + aux_cap_helper__b3,r2,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b3,r2,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b3,r2,band_HIGH_PAIRED__condX3  = 0
 c882:  5.2303475434 aux_cap_helper__b6,r7,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b6,r7,band_LOW_PAIRED__condX1
        - 1.4943850124 aux_cap_helper__b6,r7,band_LOW_PAIRED__Z2
        - aux_cap_b6,r7,band_LOW_PAIRED  = 0
 c883:  - aux_c_b7,r4 + sv_function_helpervar_b7,r4__condX1
        + sv_function_helpervar_b7,r4__condX2
        + sv_function_helpervar_b7,r4__condX3  = 0
 c884:  - 2 aux_cap_helper__b7,r3,band_UNPAIRED__Z2
        + aux_cap_helper__b7,r3,band_UNPAIRED__condX2 <= 0
 c885:  aux_cap_helper__b2,r7,band_LOW_PAIRED__Z2
        + aux_cap_helper__b2,r7,band_LOW_PAIRED__Z1  = 1
 c886:  - aux_cap_helper__b2,r9,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b2,r9,band_HIGH_PAIRED__Z2 <= 0
 c887:  125.5635947224 sv_function_helpervar_b1,r6__condX1
        + 2.1399978928 sv_function_helpervar_b1,r6__condX3
        + 5.3049280551 sv_function_helpervar_b1,r6__condX2 - aux_sv_b1,r6
        + 40.1545801822852 sv_function_helpervar_b1,r6__Z3
        + 19.2349152417227 sv_function_helpervar_b1,r6__Z2  = 0
 c888:  - 15.9946187432626 sv_function_helpervar_b4,r10__Z3
        + sv_function_helpervar_b4,r10__condX3 <= 0
 c889:  aux_cap_helper__b6,r9,band_LOW_PAIRED__condX1
        - aux_cap_helper__b6,r9,band_LOW_PAIRED__Z1 <= 0
 c890:  aux_cap_helper__b0,r0,band_LOW_PAIRED__Z2
        + aux_cap_helper__b0,r0,band_LOW_PAIRED__Z1  = 1
 c891:  - aux_cap_b5,r0,band_HIGH_PAIRED
        - 0.6181448971 aux_cap_helper__b5,r0,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b5,r0,band_HIGH_PAIRED__Z3
        + 2.1635071398 aux_cap_helper__b5,r0,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b5,r0,band_HIGH_PAIRED__condX1
        + 1.3908260184 aux_cap_helper__b5,r0,band_HIGH_PAIRED__condX3  = 0
 c892:  aux_cap_helper__b0,r4,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b0,r4,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b0,r4,band_HIGH_PAIRED__condX2
        - X__b0,r4,band_HIGH_PAIRED  = 0
 c893:  - X__b8,r5,band_LOW_PAIRED
        + aux_cap_helper__b8,r5,band_LOW_PAIRED__condX2
        + aux_cap_helper__b8,r5,band_LOW_PAIRED__condX1  = 0
 c894:  0.159946187439412 sv_function_helpervar_b2,r6__Z2
        - sv_function_helpervar_b2,r6__condX2 <= 0
 c895:  - aux_cap_helper__b5,r6,band_UNPAIRED__condX1 <= 0
 c896:  sv_function_helpervar_b0,r12__condX2
        - 5.99016947043189 sv_function_helpervar_b0,r12__Z2 <= 0
 c897:  aux_cap_helper__b8,r4,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b8,r4,band_HIGH_PAIRED__condX2 <= 0
 c898:  - sv_function_helpervar_b3,r7__condX1 <= 0
 c899:  aux_cap_helper__b9,r12,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b9,r12,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b9,r12,band_HIGH_PAIRED__condX1
        - X__b9,r12,band_HIGH_PAIRED  = 0
 c900:  - aux_cap_helper__b0,r7,band_UNPAIRED__condX1 <= 0
 c901:  aux_cap_helper__b6,r9,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b6,r9,band_LOW_PAIRED__Z2 <= 0
 c902:  2 aux_cap_helper__b8,r4,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b8,r4,band_HIGH_PAIRED__condX3 <= 0
 c903:  21.5951066722 sv_function_helpervar_b3,r9__condX3
        + 1379.5885615622 sv_function_helpervar_b3,r9__condX1
        + 66.9333385599 sv_function_helpervar_b3,r9__condX2
        + 209.954198367201 sv_function_helpervar_b3,r9__Z2
        + 471.853504591274 sv_function_helpervar_b3,r9__Z3 - aux_sv_b3,r9  = 0
 c904:  aux_cap_helper__b5,r1,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b5,r1,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b5,r1,band_HIGH_PAIRED__Z3  = 1
 c905:  X__b7,r6,band_LOW_PAIRED + X__b3,r6,band_LOW_PAIRED
        + X__b5,r6,band_LOW_PAIRED + X__b0,r6,band_LOW_PAIRED
        + X__b1,r6,band_LOW_PAIRED + X__b2,r6,band_LOW_PAIRED
        + X__b6,r6,band_LOW_PAIRED + X__b4,r6,band_LOW_PAIRED
        + X__b9,r6,band_LOW_PAIRED + X__b8,r6,band_LOW_PAIRED <= 2
 c906:  - aux_cap_helper__b2,r1,band_LOW_PAIRED__Z1
        + aux_cap_helper__b2,r1,band_LOW_PAIRED__condX1 <= 0
 c907:  aux_cap_helper__b8,r0,band_LOW_PAIRED__condX1
        - aux_cap_helper__b8,r0,band_LOW_PAIRED__Z1 <= 0
 c908:  - aux_cap_helper__b7,r3,band_UNPAIRED__Z1
        + aux_cap_helper__b7,r3,band_UNPAIRED__condX1 <= 0
 c909:  2742.33080470927 aux_sv_b8,r12 - aux_Omega_b8,r12  = 0
 c910:  - X__b1,r2,band_UNPAIRED + aux_cap_helper__b1,r2,band_UNPAIRED__condX2
        + aux_cap_helper__b1,r2,band_UNPAIRED__condX1  = 0
 c911:  - sv_function_helpervar_b8,r7__condX1 <= 0
 c912:  aux_cap_helper__b6,r2,band_UNPAIRED__Z2
        - aux_cap_helper__b6,r2,band_UNPAIRED__condX2 <= 0
 c913:  - 0.6181448971 aux_cap_helper__b0,r9,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b0,r9,band_HIGH_PAIRED__Z3
        - aux_cap_b0,r9,band_HIGH_PAIRED
        + 1.3908260184 aux_cap_helper__b0,r9,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b0,r9,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b0,r9,band_HIGH_PAIRED__condX1  = 0
 c914:  - aux_cap_helper__b4,r4,band_LOW_PAIRED__condX2
        + aux_cap_helper__b4,r4,band_LOW_PAIRED__Z2 <= 0
 c915:  3.735962531 aux_cap_helper__b0,r12,band_LOW_PAIRED__condX1
        + 5.2303475434 aux_cap_helper__b0,r12,band_LOW_PAIRED__condX2
        - aux_cap_b0,r12,band_LOW_PAIRED
        - 1.4943850124 aux_cap_helper__b0,r12,band_LOW_PAIRED__Z2  = 0
 c916:  - 2 aux_cap_helper__b2,r1,band_LOW_PAIRED__Z2
        + aux_cap_helper__b2,r1,band_LOW_PAIRED__condX2 <= 0
 c917:  sv_function_helpervar_b4,r9__condX1
        + sv_function_helpervar_b4,r9__condX3
        + sv_function_helpervar_b4,r9__condX2 - aux_c_b4,r9  = 0
 c918:  - aux_cap_helper__b1,r3,band_UNPAIRED__condX2
        + aux_cap_helper__b1,r3,band_UNPAIRED__Z2 <= 0
 c919:  aux_cap_helper__b2,r10,band_UNPAIRED__Z2
        - aux_cap_helper__b2,r10,band_UNPAIRED__condX2 <= 0
 c920:  - 3 aux_cap_helper__b1,r12,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b1,r12,band_HIGH_PAIRED__condX3 <= 0
 c921:  aux_cap_helper__b6,r6,band_LOW_PAIRED__Z1
        + aux_cap_helper__b6,r6,band_LOW_PAIRED__Z2  = 1
 c922:  - 0.159946187433409 sv_function_helpervar_b3,r5__Z1
        + sv_function_helpervar_b3,r5__condX1 <= 0
 c923:  - sv_function_helpervar_b4,r8__condX2
        + 0.159946187453367 sv_function_helpervar_b4,r8__Z2 <= 0
 c924:  aux_cap_helper__b1,r8,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b1,r8,band_HIGH_PAIRED__condX2 <= 0
 c925:  - aux_cap_helper__b2,r8,band_LOW_PAIRED__Z1
        + aux_cap_helper__b2,r8,band_LOW_PAIRED__condX1 <= 0
 c926:  sv_function_helpervar_b5,r0__condX2
        - 6.08900933282028 sv_function_helpervar_b5,r0__Z2 <= 0
 c927:  sv_function_helpervar_b7,r5__condX3
        - 15.9946187432626 sv_function_helpervar_b7,r5__Z3 <= 0
 c928:  - aux_cap_helper__b4,r4,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b4,r4,band_HIGH_PAIRED__condX1 <= 0
 c929:  aux_cap_helper__b3,r11,band_UNPAIRED__Z1
        + aux_cap_helper__b3,r11,band_UNPAIRED__Z2  = 1
 c930:  - aux_cap_helper__b9,r10,band_HIGH_PAIRED__condX1 <= 0
 c931:  aux_cap_helper__b3,r5,band_LOW_PAIRED__Z2
        - aux_cap_helper__b3,r5,band_LOW_PAIRED__condX2 <= 0
 c932:  X__b7,r10,band_HIGH_PAIRED + X__b7,r10,band_UNPAIRED
        + X__b7,r10,band_LOW_PAIRED - GlobalBidder_W_ir__b7,r10 >= 0
 c933:  aux_cap_helper__b9,r4,band_LOW_PAIRED__Z2
        - aux_cap_helper__b9,r4,band_LOW_PAIRED__condX2 <= 0
 c934:  2 aux_cap_helper__b1,r8,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b1,r8,band_HIGH_PAIRED__condX3 <= 0
 c935:  6.32727934049225 sv_function_helpervar_b4,r10__Z3
        - sv_function_helpervar_b4,r10__condX3 <= 0
 c936:  sv_function_helpervar_b5,r8__condX2
        - 6.3822247076733 sv_function_helpervar_b5,r8__Z2 <= 0
 c937:  - 15.9946187432626 sv_function_helpervar_b2,r5__Z3
        + sv_function_helpervar_b2,r5__condX3 <= 0
 c938:  aux_cap_helper__b3,r1,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b3,r1,band_HIGH_PAIRED__Z2 <= 0
 c939:  - aux_cap_b4,r1,band_LOW_PAIRED
        + 5.2303475434 aux_cap_helper__b4,r1,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b4,r1,band_LOW_PAIRED__condX1
        - 1.4943850124 aux_cap_helper__b4,r1,band_LOW_PAIRED__Z2  = 0
 c940:  - aux_sv_b4,r10 + 197.11110817987 sv_function_helpervar_b4,r10__Z2
        + 422.592773114291 sv_function_helpervar_b4,r10__Z3
        + 1289.3272634758 sv_function_helpervar_b4,r10__condX1
        + 21.3319271797 sv_function_helpervar_b4,r10__condX3
        + 56.9683599655 sv_function_helpervar_b4,r10__condX2  = 0
 c941:  0.8035888618 aux_cap_helper__b1,r8,band_UNPAIRED__condX1
        + 1.1250244065 aux_cap_helper__b1,r8,band_UNPAIRED__condX2
        - aux_cap_b1,r8,band_UNPAIRED
        - 0.3214355447 aux_cap_helper__b1,r8,band_UNPAIRED__Z2  = 0
 c942:  aux_cap_helper__b9,r0,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b9,r0,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b9,r0,band_HIGH_PAIRED__Z1  = 1
 c943:  6.78937087899278 sv_function_helpervar_b9,r2__Z3
        - sv_function_helpervar_b9,r2__condX3 <= 0
 c944:  - X__b8,r4,band_HIGH_PAIRED
        + aux_cap_helper__b8,r4,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b8,r4,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b8,r4,band_HIGH_PAIRED__condX3  = 0
 c945:  - 3 aux_cap_helper__b4,r5,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b4,r5,band_HIGH_PAIRED__condX3 <= 0
 c946:  aux_cap_helper__b5,r11,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b5,r11,band_HIGH_PAIRED__condX2 <= 0
 c947:  - aux_cap_helper__b9,r7,band_HIGH_PAIRED__condX1 <= 0
 c948:  aux_cap_helper__b7,r1,band_UNPAIRED__Z1
        + aux_cap_helper__b7,r1,band_UNPAIRED__Z2  = 1
 c949:  - sv_function_helpervar_b7,r12__condX3
        + 7.70995398967424 sv_function_helpervar_b7,r12__Z3 <= 0
 c950:  - sv_function_helpervar_b2,r6__condX1 <= 0
 c951:  - aux_cap_helper__b5,r13,band_UNPAIRED__condX2
        + aux_cap_helper__b5,r13,band_UNPAIRED__Z2 <= 0
 c952:  - sv_function_helpervar_b8,r2__condX2
        + 0.159946187437224 sv_function_helpervar_b8,r2__Z2 <= 0
 c953:  - X__b4,r11,band_LOW_PAIRED
        + aux_cap_helper__b4,r11,band_LOW_PAIRED__condX1
        + aux_cap_helper__b4,r11,band_LOW_PAIRED__condX2  = 0
 c954:  - X__b7,r6,band_UNPAIRED + aux_cap_helper__b7,r6,band_UNPAIRED__condX2
        + aux_cap_helper__b7,r6,band_UNPAIRED__condX1  = 0
 c955:  aux_cap_helper__b8,r0,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b8,r0,band_LOW_PAIRED__Z2 <= 0
 c956:  aux_cap_helper__b1,r5,band_LOW_PAIRED__condX1
        + aux_cap_helper__b1,r5,band_LOW_PAIRED__condX2
        - X__b1,r5,band_LOW_PAIRED  = 0
 c957:  X__b8,r2,band_UNPAIRED + X__b8,r2,band_LOW_PAIRED
        - GlobalBidder_W_ir__b8,r2 + X__b8,r2,band_HIGH_PAIRED >= 0
 c958:  - aux_cap_helper__b0,r11,band_LOW_PAIRED__condX2
        + aux_cap_helper__b0,r11,band_LOW_PAIRED__Z2 <= 0
 c959:  - aux_cap_helper__b3,r7,band_HIGH_PAIRED__condX1 <= 0
 c960:  - aux_cap_helper__b1,r4,band_LOW_PAIRED__condX1 <= 0
 c961:  aux_cap_helper__b7,r6,band_LOW_PAIRED__Z1
        + aux_cap_helper__b7,r6,band_LOW_PAIRED__Z2  = 1
 c962:  - aux_cap_helper__b2,r6,band_LOW_PAIRED__condX1 <= 0
 c963:  aux_cap_helper__b9,r11,band_UNPAIRED__condX1
        + aux_cap_helper__b9,r11,band_UNPAIRED__condX2
        - X__b9,r11,band_UNPAIRED  = 0
 c964:  aux_cap_helper__b4,r1,band_UNPAIRED__condX2
        + aux_cap_helper__b4,r1,band_UNPAIRED__condX1 - X__b4,r1,band_UNPAIRED
         = 0
 c965:  aux_cap_helper__b1,r9,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b1,r9,band_HIGH_PAIRED__Z2 <= 0
 c966:  - aux_cap_helper__b1,r11,band_UNPAIRED__condX1 <= 0
 c967:  - GlobalBidder_W_ir__b9,r13 + X__b9,r13,band_LOW_PAIRED
        + X__b9,r13,band_UNPAIRED + X__b9,r13,band_HIGH_PAIRED >= 0
 c968:  aux_cap_helper__b2,r1,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b2,r1,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b2,r1,band_HIGH_PAIRED__Z3  = 1
 c969:  aux_cap_b6,r10,band_LOW_PAIRED - aux_c_b6,r10
        + aux_cap_b6,r10,band_HIGH_PAIRED + aux_cap_b6,r10,band_UNPAIRED  = 0
 c970:  aux_cap_helper__b7,r7,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b7,r7,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b7,r7,band_HIGH_PAIRED__condX3
        - X__b7,r7,band_HIGH_PAIRED  = 0
 c971:  aux_cap_helper__b3,r4,band_UNPAIRED__condX1
        - aux_cap_helper__b3,r4,band_UNPAIRED__Z1 <= 0
 c972:  - aux_cap_helper__b2,r11,band_HIGH_PAIRED__condX1 <= 0
 c973:  - GlobalBidder_W_ir__b9,r13
        + 0.0102040816326531 X__b9,r13,band_LOW_PAIRED
        + 0.0102040816326531 X__b9,r13,band_UNPAIRED
        + 0.0102040816326531 X__b9,r13,band_HIGH_PAIRED <= 0
 c974:  sv_function_helpervar_b6,r0__condX3
        + sv_function_helpervar_b6,r0__condX2
        + sv_function_helpervar_b6,r0__condX1 - aux_c_b6,r0  = 0
 c975:  - aux_cap_helper__b9,r5,band_UNPAIRED__condX1 <= 0
 c976:  - aux_cap_helper__b2,r4,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b2,r4,band_HIGH_PAIRED__Z2 <= 0
 c977:  aux_cap_helper__b3,r3,band_LOW_PAIRED__condX1
        + aux_cap_helper__b3,r3,band_LOW_PAIRED__condX2
        - X__b3,r3,band_LOW_PAIRED  = 0
 c978:  aux_cap_b2,r7,band_UNPAIRED - aux_c_b2,r7
        + aux_cap_b2,r7,band_HIGH_PAIRED + aux_cap_b2,r7,band_LOW_PAIRED  = 0
 c979:  - 0.159946187556766 sv_function_helpervar_b5,r6__Z1
        + sv_function_helpervar_b5,r6__condX1 <= 0
 c980:  0.0102040816326531 X__b7,r4,band_LOW_PAIRED
        + 0.0102040816326531 X__b7,r4,band_UNPAIRED
        + 0.0102040816326531 X__b7,r4,band_HIGH_PAIRED
        - GlobalBidder_W_ir__b7,r4 <= 0
 c981:  aux_cap_helper__b9,r6,band_UNPAIRED__condX1
        + aux_cap_helper__b9,r6,band_UNPAIRED__condX2 - X__b9,r6,band_UNPAIRED
         = 0
 c982:  - sv_function_helpervar_b0,r10__condX1 <= 0
 c983:  aux_cap_helper__b6,r4,band_UNPAIRED__Z2
        + aux_cap_helper__b6,r4,band_UNPAIRED__Z1  = 1
 c984:  0.927217345653617 aux_cap_helper__b7,r2,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b7,r2,band_HIGH_PAIRED__Z2
        - aux_cap_b7,r2,band_HIGH_PAIRED
        + 1.3908260184 aux_cap_helper__b7,r2,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b7,r2,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b7,r2,band_HIGH_PAIRED__condX1  = 0
 c985:  aux_cap_helper__b8,r1,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b8,r1,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b8,r1,band_HIGH_PAIRED__Z3  = 1
 c986:  aux_cap_b3,r12,band_LOW_PAIRED + aux_cap_b3,r12,band_HIGH_PAIRED
        - aux_c_b3,r12 + aux_cap_b3,r12,band_UNPAIRED  = 0
 c987:  aux_cap_helper__b3,r4,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b3,r4,band_UNPAIRED__Z2 <= 0
 c988:  aux_cap_helper__b8,r9,band_LOW_PAIRED__condX1
        + aux_cap_helper__b8,r9,band_LOW_PAIRED__condX2
        - X__b8,r9,band_LOW_PAIRED  = 0
 c989:  0.159946187489304 sv_function_helpervar_b9,r1__Z2
        - sv_function_helpervar_b9,r1__condX2 <= 0
 c990:  - aux_cap_helper__b6,r13,band_UNPAIRED__condX1 <= 0
 c991:  - aux_cap_helper__b7,r8,band_HIGH_PAIRED__condX1 <= 0
 c992:  - 1.4943850124 aux_cap_helper__b9,r4,band_LOW_PAIRED__Z2
        - aux_cap_b9,r4,band_LOW_PAIRED
        + 5.2303475434 aux_cap_helper__b9,r4,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b9,r4,band_LOW_PAIRED__condX1  = 0
 c993:  - aux_cap_helper__b2,r4,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b2,r4,band_HIGH_PAIRED__Z3 <= 0
 c994:  sv_function_helpervar_b8,r12__Z2 + sv_function_helpervar_b8,r12__Z3
        + sv_function_helpervar_b8,r12__Z1  = 1
 c995:  - 0.3214355447 aux_cap_helper__b6,r4,band_UNPAIRED__Z2
        + 1.1250244065 aux_cap_helper__b6,r4,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b6,r4,band_UNPAIRED__condX1
        - aux_cap_b6,r4,band_UNPAIRED  = 0
 c996:  0.927217345653617 aux_cap_helper__b9,r2,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b9,r2,band_HIGH_PAIRED__Z2
        - aux_cap_b9,r2,band_HIGH_PAIRED
        + 2.1635071398 aux_cap_helper__b9,r2,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b9,r2,band_HIGH_PAIRED__condX1
        + 1.3908260184 aux_cap_helper__b9,r2,band_HIGH_PAIRED__condX3  = 0
 c997:  1.1250244065 aux_cap_helper__b0,r0,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b0,r0,band_UNPAIRED__condX1
        - aux_cap_b0,r0,band_UNPAIRED
        - 0.3214355447 aux_cap_helper__b0,r0,band_UNPAIRED__Z2  = 0
 c998:  - aux_cap_helper__b8,r2,band_LOW_PAIRED__condX1 <= 0
 c999:  2 aux_cap_helper__b5,r11,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b5,r11,band_HIGH_PAIRED__condX3 <= 0
 c1000: - 3 aux_cap_helper__b8,r13,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b8,r13,band_HIGH_PAIRED__condX3 <= 0
 c1001: 0.0102040816326531 X__b8,r2,band_UNPAIRED
        + 0.0102040816326531 X__b8,r2,band_LOW_PAIRED
        - GlobalBidder_W_ir__b8,r2
        + 0.0102040816326531 X__b8,r2,band_HIGH_PAIRED <= 0
 c1002: aux_cap_helper__b0,r6,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b0,r6,band_HIGH_PAIRED__Z2 <= 0
 c1003: aux_cap_helper__b3,r5,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b3,r5,band_UNPAIRED__Z2 <= 0
 c1004: - aux_cap_b0,r12,band_UNPAIRED
        - 0.3214355447 aux_cap_helper__b0,r12,band_UNPAIRED__Z2
        + 0.8035888618 aux_cap_helper__b0,r12,band_UNPAIRED__condX1
        + 1.1250244065 aux_cap_helper__b0,r12,band_UNPAIRED__condX2  = 0
 c1005: aux_cap_b4,r3,band_LOW_PAIRED + aux_cap_b4,r3,band_UNPAIRED
        + aux_cap_b4,r3,band_HIGH_PAIRED - aux_c_b4,r3  = 0
 c1006: - sv_function_helpervar_b6,r12__condX3
        + 5.83628230872818 sv_function_helpervar_b6,r12__Z3 <= 0
 c1007: - aux_cap_helper__b7,r6,band_LOW_PAIRED__condX1 <= 0
 c1008: 16.8296389080731 sv_function_helpervar_b0,r11__Z2
        + 35.1719302955087 sv_function_helpervar_b0,r11__Z3
        + 4.6500307338 sv_function_helpervar_b0,r11__condX2
        + 109.8706626077 sv_function_helpervar_b0,r11__condX1
        + 1.8702985755 sv_function_helpervar_b0,r11__condX3 - aux_sv_b0,r11
         = 0
 c1009: sv_function_helpervar_b1,r7__condX2
        - 6.61140040786873 sv_function_helpervar_b1,r7__Z2 <= 0
 c1010: aux_cap_helper__b1,r10,band_LOW_PAIRED__Z2
        - aux_cap_helper__b1,r10,band_LOW_PAIRED__condX2 <= 0
 c1011: sv_function_helpervar_b1,r4__condX3
        - 15.9946187432626 sv_function_helpervar_b1,r4__Z3 <= 0
 c1012: sv_function_helpervar_b9,r8__Z1 + sv_function_helpervar_b9,r8__Z2
        + sv_function_helpervar_b9,r8__Z3  = 1
 c1013: - 2 aux_cap_helper__b2,r8,band_LOW_PAIRED__Z2
        + aux_cap_helper__b2,r8,band_LOW_PAIRED__condX2 <= 0
 c1014: - aux_cap_helper__b7,r2,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b7,r2,band_HIGH_PAIRED__condX1 <= 0
 c1015: aux_cap_helper__b1,r11,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b1,r11,band_HIGH_PAIRED__Z2 <= 0
 c1016: - aux_cap_helper__b3,r0,band_HIGH_PAIRED__condX1 <= 0
 c1017: aux_cap_helper__b1,r2,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b1,r2,band_HIGH_PAIRED__Z1 <= 0
 c1018: aux_cap_helper__b1,r1,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b1,r1,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b1,r1,band_HIGH_PAIRED__Z1  = 1
 c1019: aux_cap_helper__b2,r13,band_UNPAIRED__Z1
        + aux_cap_helper__b2,r13,band_UNPAIRED__Z2  = 1
 c1020: GlobalBidder_PSI__i7,k2 - 18497526.8940961 GlobalBidder_hat{W}__i7,k2
        <= 0
 c1021: - 3 aux_cap_helper__b5,r8,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b5,r8,band_HIGH_PAIRED__condX3 <= 0
 c1022: - aux_cap_helper__b7,r13,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b7,r13,band_HIGH_PAIRED__Z2 <= 0
 c1023: - X__b2,r5,band_HIGH_PAIRED
        + aux_cap_helper__b2,r5,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b2,r5,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b2,r5,band_HIGH_PAIRED__condX3  = 0
 c1024: - 2 aux_cap_helper__b7,r2,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b7,r2,band_HIGH_PAIRED__condX2 <= 0
 c1025: 32.4379228422 sv_function_helpervar_b7,r13__condX3
        + 75.1323026501 sv_function_helpervar_b7,r13__condX2
        + 1850.3233245889 sv_function_helpervar_b7,r13__condX1
        + 283.935036138617 sv_function_helpervar_b7,r13__Z2
        + 577.286908018701 sv_function_helpervar_b7,r13__Z3 - aux_sv_b7,r13
         = 0
 c1026: - sv_function_helpervar_b6,r1__condX1 <= 0
 c1027: sv_function_helpervar_b5,r1__condX2
        + sv_function_helpervar_b5,r1__condX1
        + sv_function_helpervar_b5,r1__condX3 - aux_c_b5,r1  = 0
 c1028: aux_cap_helper__b1,r2,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b1,r2,band_HIGH_PAIRED__Z2 <= 0
 c1029: - aux_cap_helper__b9,r4,band_UNPAIRED__condX1 <= 0
 c1030: 0.0102040816326531 X__b9,r8,band_HIGH_PAIRED - GlobalBidder_W_ir__b9,r8
        + 0.0102040816326531 X__b9,r8,band_UNPAIRED
        + 0.0102040816326531 X__b9,r8,band_LOW_PAIRED <= 0
 c1031: aux_cap_b9,r4,band_HIGH_PAIRED + aux_cap_b9,r4,band_UNPAIRED
        + aux_cap_b9,r4,band_LOW_PAIRED - aux_c_b9,r4  = 0
 c1032: aux_cap_b0,r5,band_LOW_PAIRED - aux_c_b0,r5
        + aux_cap_b0,r5,band_UNPAIRED + aux_cap_b0,r5,band_HIGH_PAIRED  = 0
 c1033: 0.0102040816326531 X__b9,r0,band_HIGH_PAIRED
        + 0.0102040816326531 X__b9,r0,band_LOW_PAIRED
        - GlobalBidder_W_ir__b9,r0 + 0.0102040816326531 X__b9,r0,band_UNPAIRED
        <= 0
 c1034: aux_cap_b3,r5,band_UNPAIRED + aux_cap_b3,r5,band_LOW_PAIRED
        - aux_c_b3,r5 + aux_cap_b3,r5,band_HIGH_PAIRED  = 0
 c1035: - aux_cap_helper__b6,r12,band_UNPAIRED__condX1 <= 0
 c1036: - aux_cap_helper__b9,r0,band_HIGH_PAIRED__condX1 <= 0
 c1037: sv_function_helpervar_b1,r2__Z1 + sv_function_helpervar_b1,r2__Z3
        + sv_function_helpervar_b1,r2__Z2  = 1
 c1038: - X__b5,r6,band_HIGH_PAIRED
        + aux_cap_helper__b5,r6,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b5,r6,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b5,r6,band_HIGH_PAIRED__condX1  = 0
 c1039: - sv_function_helpervar_b8,r4__condX2
        + 0.159946187488937 sv_function_helpervar_b8,r4__Z2 <= 0
 c1040: - 18497526.8940961 GlobalBidder_hat{W}__i8,k1 - GlobalBidder_PSI__i8,k1
        + aux_Omega_b8,r12 + aux_Omega_b8,r13 + aux_Omega_b8,r10
        + aux_Omega_b8,r11 + aux_Omega_b8,r5 + aux_Omega_b8,r6
        + aux_Omega_b8,r7 + aux_Omega_b8,r8 + aux_Omega_b8,r9 + aux_Omega_b8,r0
        + aux_Omega_b8,r1 + aux_Omega_b8,r2 + aux_Omega_b8,r3 + aux_Omega_b8,r4
        >= -18497526.8940961
 c1041: aux_cap_helper__b0,r6,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b0,r6,band_HIGH_PAIRED__Z1 <= 0
 c1042: aux_cap_helper__b3,r5,band_UNPAIRED__condX1
        - aux_cap_helper__b3,r5,band_UNPAIRED__Z1 <= 0
 c1043: 42.976244427678 sv_function_helpervar_b1,r1__Z3
        + 19.1073156122706 sv_function_helpervar_b1,r1__Z2
        + 6.1026940501 sv_function_helpervar_b1,r1__condX2
        + 125.5635947147 sv_function_helpervar_b1,r1__condX1
        + 1.9635845446 sv_function_helpervar_b1,r1__condX3 - aux_sv_b1,r1  = 0
 c1044: aux_cap_helper__b3,r7,band_LOW_PAIRED__Z2
        + aux_cap_helper__b3,r7,band_LOW_PAIRED__Z1  = 1
 c1045: aux_cap_helper__b1,r11,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b1,r11,band_HIGH_PAIRED__Z1 <= 0
 c1046: 0.15994618747751 sv_function_helpervar_b3,r2__Z2
        - sv_function_helpervar_b3,r2__condX2 <= 0
 c1047: - aux_cap_helper__b8,r8,band_UNPAIRED__condX1 <= 0
 c1048: aux_cap_helper__b6,r7,band_UNPAIRED__condX2
        + aux_cap_helper__b6,r7,band_UNPAIRED__condX1 - X__b6,r7,band_UNPAIRED
         = 0
 c1049: 0.0102040816326531 X__b9,r10,band_LOW_PAIRED
        + 0.0102040816326531 X__b9,r10,band_UNPAIRED
        - GlobalBidder_W_ir__b9,r10
        + 0.0102040816326531 X__b9,r10,band_HIGH_PAIRED <= 0
 c1050: sv_function_helpervar_b3,r12__Z1 + sv_function_helpervar_b3,r12__Z2
        + sv_function_helpervar_b3,r12__Z3  = 1
 c1051: - 2 aux_cap_helper__b9,r1,band_UNPAIRED__Z2
        + aux_cap_helper__b9,r1,band_UNPAIRED__condX2 <= 0
 c1052: - aux_cap_helper__b4,r3,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b4,r3,band_HIGH_PAIRED__Z2 <= 0
 c1053: - aux_cap_helper__b9,r1,band_UNPAIRED__Z1
        + aux_cap_helper__b9,r1,band_UNPAIRED__condX1 <= 0
 c1054: - aux_cap_helper__b2,r5,band_UNPAIRED__condX1 <= 0
 c1055: - aux_cap_helper__b4,r4,band_UNPAIRED__condX2
        + aux_cap_helper__b4,r4,band_UNPAIRED__Z2 <= 0
 c1056: - aux_cap_b2,r12,band_UNPAIRED
        + 0.8035888618 aux_cap_helper__b2,r12,band_UNPAIRED__condX1
        + 1.1250244065 aux_cap_helper__b2,r12,band_UNPAIRED__condX2
        - 0.3214355447 aux_cap_helper__b2,r12,band_UNPAIRED__Z2  = 0
 c1057: - 3 aux_cap_helper__b5,r13,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b5,r13,band_HIGH_PAIRED__condX3 <= 0
 c1058: - aux_cap_helper__b3,r8,band_UNPAIRED__condX1 <= 0
 c1059: 451.097625545873 sv_function_helpervar_b3,r6__Z3
        + 210.955012738728 sv_function_helpervar_b3,r6__Z2
        + 1379.5885616453 sv_function_helpervar_b3,r6__condX1
        + 60.6761442695 sv_function_helpervar_b3,r6__condX2
        + 22.892785559 sv_function_helpervar_b3,r6__condX3 - aux_sv_b3,r6  = 0
 c1060: - aux_cap_helper__b4,r3,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b4,r3,band_HIGH_PAIRED__Z3 <= 0
 c1061: aux_cap_helper__b5,r7,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b5,r7,band_LOW_PAIRED__Z2 <= 0
 c1062: 2.1635071398 aux_cap_helper__b4,r8,band_HIGH_PAIRED__condX2
        + 1.3908260184 aux_cap_helper__b4,r8,band_HIGH_PAIRED__condX3
        + 1.5453622427 aux_cap_helper__b4,r8,band_HIGH_PAIRED__condX1
        + 0.927217345653617 aux_cap_helper__b4,r8,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b4,r8,band_HIGH_PAIRED__Z2
        - aux_cap_b4,r8,band_HIGH_PAIRED  = 0
 c1063: - aux_Omega_b4,r9 + 202.180870392989 aux_sv_b4,r9  = 0
 c1064: sv_function_helpervar_b1,r4__Z3 + sv_function_helpervar_b1,r4__Z2
        + sv_function_helpervar_b1,r4__Z1  = 1
 c1065: aux_cap_helper__b5,r10,band_LOW_PAIRED__condX1
        + aux_cap_helper__b5,r10,band_LOW_PAIRED__condX2
        - X__b5,r10,band_LOW_PAIRED  = 0
 c1066: - aux_c_b7,r11 + sv_function_helpervar_b7,r11__condX3
        + sv_function_helpervar_b7,r11__condX2
        + sv_function_helpervar_b7,r11__condX1  = 0
 c1067: - aux_cap_helper__b9,r2,band_LOW_PAIRED__Z1
        + aux_cap_helper__b9,r2,band_LOW_PAIRED__condX1 <= 0
 c1068: aux_cap_helper__b3,r13,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b3,r13,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b3,r13,band_HIGH_PAIRED__Z3  = 1
 c1069: 5.89050642741838 sv_function_helpervar_b4,r11__Z3
        - sv_function_helpervar_b4,r11__condX3 <= 0
 c1070: sv_function_helpervar_b6,r13__condX1
        + sv_function_helpervar_b6,r13__condX2
        + sv_function_helpervar_b6,r13__condX3 - aux_c_b6,r13  = 0
 c1071: - aux_cap_helper__b1,r2,band_UNPAIRED__condX2
        + aux_cap_helper__b1,r2,band_UNPAIRED__Z2 <= 0
 c1072: sv_function_helpervar_b5,r9__Z1 + sv_function_helpervar_b5,r9__Z2
        + sv_function_helpervar_b5,r9__Z3  = 1
 c1073: 6.30423233120729 sv_function_helpervar_b0,r4__Z3
        - sv_function_helpervar_b0,r4__condX3 <= 0
 c1074: aux_cap_helper__b3,r8,band_UNPAIRED__condX2
        + aux_cap_helper__b3,r8,band_UNPAIRED__condX1 - X__b3,r8,band_UNPAIRED
         = 0
 c1075: - aux_cap_helper__b0,r6,band_UNPAIRED__condX1 <= 0
 c1076: sv_function_helpervar_b1,r2__condX2
        - 5.77334678997269 sv_function_helpervar_b1,r2__Z2 <= 0
 c1077: sv_function_helpervar_b6,r9__condX1
        - 0.159946187505012 sv_function_helpervar_b6,r9__Z1 <= 0
 c1078: - 0.159946187460974 sv_function_helpervar_b2,r12__Z1
        + sv_function_helpervar_b2,r12__condX1 <= 0
 c1079: aux_cap_helper__b4,r7,band_UNPAIRED__Z2
        + aux_cap_helper__b4,r7,band_UNPAIRED__Z1  = 1
 c1080: 414.868336224797 aux_sv_b5,r10 - aux_Omega_b5,r10  = 0
 c1081: - X__b5,r0,band_UNPAIRED + aux_cap_helper__b5,r0,band_UNPAIRED__condX2
        + aux_cap_helper__b5,r0,band_UNPAIRED__condX1  = 0
 c1082: - 2 aux_cap_helper__b9,r2,band_LOW_PAIRED__Z2
        + aux_cap_helper__b9,r2,band_LOW_PAIRED__condX2 <= 0
 c1083: - aux_cap_helper__b7,r13,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b7,r13,band_HIGH_PAIRED__Z3 <= 0
 c1084: sv_function_helpervar_b5,r8__condX3
        - 15.9946187432626 sv_function_helpervar_b5,r8__Z3 <= 0
 c1085: - aux_cap_helper__b6,r6,band_LOW_PAIRED__Z1
        + aux_cap_helper__b6,r6,band_LOW_PAIRED__condX1 <= 0
 c1086: aux_cap_helper__b3,r1,band_UNPAIRED__Z2
        - aux_cap_helper__b3,r1,band_UNPAIRED__condX2 <= 0
 c1087: - 2 aux_cap_helper__b7,r9,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b7,r9,band_HIGH_PAIRED__condX2 <= 0
 c1088: 0.8035888618 aux_cap_helper__b0,r4,band_UNPAIRED__condX1
        + 1.1250244065 aux_cap_helper__b0,r4,band_UNPAIRED__condX2
        - aux_cap_b0,r4,band_UNPAIRED
        - 0.3214355447 aux_cap_helper__b0,r4,band_UNPAIRED__Z2  = 0
 c1089: aux_cap_helper__b2,r8,band_LOW_PAIRED__Z2
        + aux_cap_helper__b2,r8,band_LOW_PAIRED__Z1  = 1
 c1090: 0.159946187434666 sv_function_helpervar_b7,r8__Z2
        - sv_function_helpervar_b7,r8__condX2 <= 0
 c1091: aux_cap_helper__b7,r2,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b7,r2,band_UNPAIRED__Z2 <= 0
 c1092: - aux_cap_helper__b9,r12,band_UNPAIRED__Z1
        + aux_cap_helper__b9,r12,band_UNPAIRED__condX1 <= 0
 c1093: aux_cap_helper__b0,r0,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b0,r0,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b0,r0,band_HIGH_PAIRED__Z3  = 1
 c1094: aux_cap_helper__b1,r8,band_UNPAIRED__Z1
        + aux_cap_helper__b1,r8,band_UNPAIRED__Z2  = 1
 c1095: aux_cap_helper__b4,r9,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b4,r9,band_HIGH_PAIRED__Z2 <= 0
 c1096: - aux_cap_helper__b1,r9,band_UNPAIRED__condX1 <= 0
 c1097: aux_cap_helper__b4,r7,band_LOW_PAIRED__Z2
        - aux_cap_helper__b4,r7,band_LOW_PAIRED__condX2 <= 0
 c1098: 6.67344018864281 sv_function_helpervar_b1,r8__Z3
        - sv_function_helpervar_b1,r8__condX3 <= 0
 c1099: - aux_cap_b9,r11,band_HIGH_PAIRED
        + 2.1635071398 aux_cap_helper__b9,r11,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b9,r11,band_HIGH_PAIRED__condX1
        + 1.3908260184 aux_cap_helper__b9,r11,band_HIGH_PAIRED__condX3
        - 0.6181448971 aux_cap_helper__b9,r11,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b9,r11,band_HIGH_PAIRED__Z3  = 0
 c1100: - aux_cap_helper__b1,r10,band_HIGH_PAIRED__condX1 <= 0
 c1101: 2 aux_cap_helper__b7,r3,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b7,r3,band_HIGH_PAIRED__condX3 <= 0
 c1102: - sv_function_helpervar_b9,r13__condX1 <= 0
 c1103: 5.60347911969823 sv_function_helpervar_b4,r1__Z3
        - sv_function_helpervar_b4,r1__condX3 <= 0
 c1104: - 2 aux_cap_helper__b9,r12,band_UNPAIRED__Z2
        + aux_cap_helper__b9,r12,band_UNPAIRED__condX2 <= 0
 c1105: - 0.159946187462481 sv_function_helpervar_b0,r8__Z1
        + sv_function_helpervar_b0,r8__condX1 <= 0
 c1106: - aux_cap_b1,r12,band_UNPAIRED
        + 1.1250244065 aux_cap_helper__b1,r12,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b1,r12,band_UNPAIRED__condX1
        - 0.3214355447 aux_cap_helper__b1,r12,band_UNPAIRED__Z2  = 0
 c1107: aux_cap_helper__b3,r13,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b3,r13,band_HIGH_PAIRED__Z2 <= 0
 c1108: - 3 aux_cap_helper__b1,r0,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b1,r0,band_HIGH_PAIRED__condX3 <= 0
 c1109: - aux_cap_helper__b1,r3,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b1,r3,band_HIGH_PAIRED__Z2 <= 0
 c1110: 5.1352845855 sv_function_helpervar_b0,r12__condX2
        + 1.7565578191 sv_function_helpervar_b0,r12__condX3
        + 109.8706626024 sv_function_helpervar_b0,r12__condX1
        + 16.7520244045212 sv_function_helpervar_b0,r12__Z2
        + 36.9911703295265 sv_function_helpervar_b0,r12__Z3 - aux_sv_b0,r12
         = 0
 c1111: aux_cap_helper__b8,r11,band_UNPAIRED__Z2
        + aux_cap_helper__b8,r11,band_UNPAIRED__Z1  = 1
 c1112: - sv_function_helpervar_b5,r0__condX1 <= 0
 c1113: 0.927217345653617 aux_cap_helper__b3,r3,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b3,r3,band_HIGH_PAIRED__Z2
        - aux_cap_b3,r3,band_HIGH_PAIRED
        + 1.3908260184 aux_cap_helper__b3,r3,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b3,r3,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b3,r3,band_HIGH_PAIRED__condX1  = 0
 c1114: - 0.159946187525574 sv_function_helpervar_b0,r1__Z1
        + sv_function_helpervar_b0,r1__condX1 <= 0
 c1115: sv_function_helpervar_b5,r7__Z3 + sv_function_helpervar_b5,r7__Z1
        + sv_function_helpervar_b5,r7__Z2  = 1
 c1116: - 5.77782566925271 sv_function_helpervar_b3,r4__Z2
        + sv_function_helpervar_b3,r4__condX2 <= 0
 c1117: aux_cap_helper__b6,r3,band_UNPAIRED__Z2
        - aux_cap_helper__b6,r3,band_UNPAIRED__condX2 <= 0
 c1118: - aux_cap_helper__b7,r9,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b7,r9,band_HIGH_PAIRED__condX1 <= 0
 c1119: sv_function_helpervar_b2,r10__condX3
        - 15.9946187432626 sv_function_helpervar_b2,r10__Z3 <= 0
 c1120: - 3 aux_cap_helper__b7,r0,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b7,r0,band_HIGH_PAIRED__condX3 <= 0
 c1121: - sv_function_helpervar_b9,r4__condX2
        + 0.159946187477637 sv_function_helpervar_b9,r4__Z2 <= 0
 c1122: - aux_cap_helper__b1,r3,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b1,r3,band_HIGH_PAIRED__Z3 <= 0
 c1123: - 1.4943850124 aux_cap_helper__b3,r3,band_LOW_PAIRED__Z2
        - aux_cap_b3,r3,band_LOW_PAIRED
        + 3.735962531 aux_cap_helper__b3,r3,band_LOW_PAIRED__condX1
        + 5.2303475434 aux_cap_helper__b3,r3,band_LOW_PAIRED__condX2  = 0
 c1124: aux_cap_helper__b7,r2,band_UNPAIRED__condX1
        - aux_cap_helper__b7,r2,band_UNPAIRED__Z1 <= 0
 c1125: aux_cap_helper__b4,r9,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b4,r9,band_HIGH_PAIRED__Z1 <= 0
 c1126: aux_cap_b5,r1,band_HIGH_PAIRED - aux_c_b5,r1
        + aux_cap_b5,r1,band_UNPAIRED + aux_cap_b5,r1,band_LOW_PAIRED  = 0
 c1127: aux_cap_helper__b7,r11,band_UNPAIRED__Z2
        - aux_cap_helper__b7,r11,band_UNPAIRED__condX2 <= 0
 c1128: aux_cap_helper__b4,r0,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b4,r0,band_HIGH_PAIRED__Z3 <= 0
 c1129: - X__b5,r10,band_HIGH_PAIRED
        + aux_cap_helper__b5,r10,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b5,r10,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b5,r10,band_HIGH_PAIRED__condX2  = 0
 c1130: - aux_cap_helper__b0,r7,band_HIGH_PAIRED__condX1 <= 0
 c1131: aux_cap_helper__b7,r3,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b7,r3,band_HIGH_PAIRED__condX2 <= 0
 c1132: aux_cap_helper__b1,r9,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b1,r9,band_HIGH_PAIRED__Z1 <= 0
 c1133: sv_function_helpervar_b8,r12__condX3
        - 15.9946187432626 sv_function_helpervar_b8,r12__Z3 <= 0
 c1134: - aux_cap_helper__b5,r7,band_UNPAIRED__condX1 <= 0
 c1135: - 2 aux_cap_helper__b6,r6,band_LOW_PAIRED__Z2
        + aux_cap_helper__b6,r6,band_LOW_PAIRED__condX2 <= 0
 c1136: - sv_function_helpervar_b0,r9__condX1 <= 0
 c1137: aux_cap_helper__b6,r6,band_UNPAIRED__condX1
        - aux_cap_helper__b6,r6,band_UNPAIRED__Z1 <= 0
 c1138: aux_cap_helper__b7,r5,band_LOW_PAIRED__Z2
        + aux_cap_helper__b7,r5,band_LOW_PAIRED__Z1  = 1
 c1139: - sv_function_helpervar_b9,r11__condX3
        + 7.76584458655766 sv_function_helpervar_b9,r11__Z3 <= 0
 c1140: - GlobalBidder_W_i_8 + GlobalBidder_W_ir__b8,r10
        + GlobalBidder_W_ir__b8,r11 + GlobalBidder_W_ir__b8,r12
        + GlobalBidder_W_ir__b8,r13 + GlobalBidder_W_ir__b8,r8
        + GlobalBidder_W_ir__b8,r7 + GlobalBidder_W_ir__b8,r6
        + GlobalBidder_W_ir__b8,r5 + GlobalBidder_W_ir__b8,r9
        + GlobalBidder_W_ir__b8,r0 + GlobalBidder_W_ir__b8,r4
        + GlobalBidder_W_ir__b8,r3 + GlobalBidder_W_ir__b8,r2
        + GlobalBidder_W_ir__b8,r1  = 0
 c1141: aux_cap_helper__b3,r13,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b3,r13,band_HIGH_PAIRED__Z1 <= 0
 c1142: - 15.9946187432626 sv_function_helpervar_b0,r8__Z3
        + sv_function_helpervar_b0,r8__condX3 <= 0
 c1143: aux_cap_helper__b9,r7,band_LOW_PAIRED__Z2
        - aux_cap_helper__b9,r7,band_LOW_PAIRED__condX2 <= 0
 c1144: - aux_cap_helper__b8,r5,band_LOW_PAIRED__condX1 <= 0
 c1145: - 2 aux_cap_helper__b0,r3,band_UNPAIRED__Z2
        + aux_cap_helper__b0,r3,band_UNPAIRED__condX2 <= 0
 c1146: - aux_cap_helper__b7,r3,band_LOW_PAIRED__condX1 <= 0
 c1147: - aux_cap_helper__b0,r3,band_UNPAIRED__Z1
        + aux_cap_helper__b0,r3,band_UNPAIRED__condX1 <= 0
 c1148: - 6.16837738919705 sv_function_helpervar_b0,r8__Z2
        + sv_function_helpervar_b0,r8__condX2 <= 0
 c1149: - X__b1,r12,band_LOW_PAIRED
        + aux_cap_helper__b1,r12,band_LOW_PAIRED__condX1
        + aux_cap_helper__b1,r12,band_LOW_PAIRED__condX2  = 0
 c1150: aux_cap_helper__b6,r6,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b6,r6,band_UNPAIRED__Z2 <= 0
 c1151: aux_cap_helper__b9,r8,band_UNPAIRED__Z1
        + aux_cap_helper__b9,r8,band_UNPAIRED__Z2  = 1
 c1152: 17.3029165395 sv_function_helpervar_b6,r9__condX3
        + 47.7909877731 sv_function_helpervar_b6,r9__condX2
        + 1059.4718671056 sv_function_helpervar_b6,r9__condX1
        + 350.870468703236 sv_function_helpervar_b6,r9__Z3
        + 161.814499620952 sv_function_helpervar_b6,r9__Z2 - aux_sv_b6,r9  = 0
 c1153: - aux_Omega_b9,r6 + 50.3736070261399 aux_sv_b9,r6  = 0
 c1154: 3.735962531 aux_cap_helper__b3,r1,band_LOW_PAIRED__condX1
        + 5.2303475434 aux_cap_helper__b3,r1,band_LOW_PAIRED__condX2
        - aux_cap_b3,r1,band_LOW_PAIRED
        - 1.4943850124 aux_cap_helper__b3,r1,band_LOW_PAIRED__Z2  = 0
 c1155: sv_function_helpervar_b8,r11__condX2
        - 7.71565157693315 sv_function_helpervar_b8,r11__Z2 <= 0
 c1156: - aux_cap_helper__b0,r1,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b0,r1,band_HIGH_PAIRED__condX1 <= 0
 c1157: - 1.4943850124 aux_cap_helper__b1,r13,band_LOW_PAIRED__Z2
        + 3.735962531 aux_cap_helper__b1,r13,band_LOW_PAIRED__condX1
        + 5.2303475434 aux_cap_helper__b1,r13,band_LOW_PAIRED__condX2
        - aux_cap_b1,r13,band_LOW_PAIRED  = 0
 c1158: - sv_function_helpervar_b2,r11__condX1 <= 0
 c1159: aux_cap_helper__b2,r0,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b2,r0,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b2,r0,band_HIGH_PAIRED__Z1  = 1
 c1160: - aux_sv_b5,r4 + 18.1647959554 sv_function_helpervar_b5,r4__condX3
        + 57.5712356876 sv_function_helpervar_b5,r4__condX2
        + 1169.5974424101 sv_function_helpervar_b5,r4__condX1
        + 402.322687288737 sv_function_helpervar_b5,r4__Z3
        + 177.864352094705 sv_function_helpervar_b5,r4__Z2  = 0
 c1161: aux_cap_helper__b0,r2,band_UNPAIRED__Z2
        + aux_cap_helper__b0,r2,band_UNPAIRED__Z1  = 1
 c1162: sv_function_helpervar_b1,r7__condX2
        + sv_function_helpervar_b1,r7__condX1
        + sv_function_helpervar_b1,r7__condX3 - aux_c_b1,r7  = 0
 c1163: aux_cap_helper__b7,r13,band_UNPAIRED__Z2
        + aux_cap_helper__b7,r13,band_UNPAIRED__Z1  = 1
 c1164: - aux_cap_helper__b1,r1,band_LOW_PAIRED__condX1 <= 0
 c1165: - aux_cap_b3,r2,band_UNPAIRED
        + 1.1250244065 aux_cap_helper__b3,r2,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b3,r2,band_UNPAIRED__condX1
        - 0.3214355447 aux_cap_helper__b3,r2,band_UNPAIRED__Z2  = 0
 c1166: X__b7,r4,band_LOW_PAIRED + X__b7,r4,band_UNPAIRED
        + X__b7,r4,band_HIGH_PAIRED - GlobalBidder_W_ir__b7,r4 >= 0
 c1167: 1.5453622427 aux_cap_helper__b1,r11,band_HIGH_PAIRED__condX1
        + 2.1635071398 aux_cap_helper__b1,r11,band_HIGH_PAIRED__condX2
        + 1.3908260184 aux_cap_helper__b1,r11,band_HIGH_PAIRED__condX3
        + 0.927217345653617 aux_cap_helper__b1,r11,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b1,r11,band_HIGH_PAIRED__Z2
        - aux_cap_b1,r11,band_HIGH_PAIRED  = 0
 c1168: - 2 aux_cap_helper__b0,r1,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b0,r1,band_HIGH_PAIRED__condX2 <= 0
 c1169: - sv_function_helpervar_b3,r8__condX3
        + 5.93313076510357 sv_function_helpervar_b3,r8__Z3 <= 0
 c1170: aux_cap_helper__b4,r11,band_LOW_PAIRED__Z2
        + aux_cap_helper__b4,r11,band_LOW_PAIRED__Z1  = 1
 c1171: sv_function_helpervar_b5,r3__Z2 + sv_function_helpervar_b5,r3__Z1
        + sv_function_helpervar_b5,r3__Z3  = 1
 c1172: - aux_cap_helper__b6,r12,band_HIGH_PAIRED__condX1 <= 0
 c1173: aux_cap_helper__b7,r0,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b7,r0,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b7,r0,band_HIGH_PAIRED__Z1  = 1
 c1174: aux_cap_helper__b5,r0,band_UNPAIRED__Z2
        - aux_cap_helper__b5,r0,band_UNPAIRED__condX2 <= 0
 c1175: aux_cap_helper__b6,r1,band_LOW_PAIRED__Z2
        - aux_cap_helper__b6,r1,band_LOW_PAIRED__condX2 <= 0
 c1176: aux_cap_helper__b9,r4,band_LOW_PAIRED__condX2
        + aux_cap_helper__b9,r4,band_LOW_PAIRED__condX1
        - X__b9,r4,band_LOW_PAIRED  = 0
 c1177: aux_cap_helper__b5,r3,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b5,r3,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b5,r3,band_HIGH_PAIRED__condX2
        - X__b5,r3,band_HIGH_PAIRED  = 0
 c1178: - sv_function_helpervar_b7,r6__condX1 <= 0
 c1179: - aux_Omega_b3,r6 + 55.9717077637061 aux_sv_b3,r6  = 0
 c1180: aux_cap_helper__b6,r12,band_LOW_PAIRED__condX2
        + aux_cap_helper__b6,r12,band_LOW_PAIRED__condX1
        - X__b6,r12,band_LOW_PAIRED  = 0
 c1181: - sv_function_helpervar_b0,r0__condX1 <= 0
 c1182: aux_cap_helper__b6,r0,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b6,r0,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b6,r0,band_HIGH_PAIRED__Z1  = 1
 c1183: - 0.159946187479682 sv_function_helpervar_b2,r5__Z1
        + sv_function_helpervar_b2,r5__condX1 <= 0
 c1184: aux_cap_helper__b3,r1,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b3,r1,band_HIGH_PAIRED__Z1 <= 0
 c1185: - aux_cap_helper__b8,r5,band_UNPAIRED__Z1
        + aux_cap_helper__b8,r5,band_UNPAIRED__condX1 <= 0
 c1186: - X__b0,r1,band_HIGH_PAIRED
        + aux_cap_helper__b0,r1,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b0,r1,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b0,r1,band_HIGH_PAIRED__condX3  = 0
 c1187: sv_function_helpervar_b8,r6__condX3
        - 15.9946187432626 sv_function_helpervar_b8,r6__Z3 <= 0
 c1188: - X__b7,r6,band_LOW_PAIRED
        + aux_cap_helper__b7,r6,band_LOW_PAIRED__condX1
        + aux_cap_helper__b7,r6,band_LOW_PAIRED__condX2  = 0
 c1189: aux_cap_helper__b5,r3,band_UNPAIRED__Z2
        + aux_cap_helper__b5,r3,band_UNPAIRED__Z1  = 1
 c1190: - aux_Omega_b1,r7 + 3455.69657342048 aux_sv_b1,r7  = 0
 c1191: - 7.89939018985145 sv_function_helpervar_b8,r13__Z2
        + sv_function_helpervar_b8,r13__condX2 <= 0
 c1192: - 0.159946187491808 sv_function_helpervar_b9,r12__Z1
        + sv_function_helpervar_b9,r12__condX1 <= 0
 c1193: - aux_Omega_b3,r7 + 2752.52715680771 aux_sv_b3,r7  = 0
 c1194: sv_function_helpervar_b1,r8__Z3 + sv_function_helpervar_b1,r8__Z2
        + sv_function_helpervar_b1,r8__Z1  = 1
 c1195: - aux_cap_helper__b8,r10,band_UNPAIRED__condX1 <= 0
 c1196: - 2 aux_cap_helper__b8,r5,band_UNPAIRED__Z2
        + aux_cap_helper__b8,r5,band_UNPAIRED__condX2 <= 0
 c1197: aux_Omega_b7,r10 + aux_Omega_b7,r11 + aux_Omega_b7,r12
        + aux_Omega_b7,r13 + aux_Omega_b7,r0 + aux_Omega_b7,r1
        + aux_Omega_b7,r2 + aux_Omega_b7,r3 + aux_Omega_b7,r4 + aux_Omega_b7,r5
        + aux_Omega_b7,r6 + aux_Omega_b7,r7 + aux_Omega_b7,r8 + aux_Omega_b7,r9
        - GlobalBidder_PSI__i7,k1 + 18497526.8940961 GlobalBidder_hat{W}__i7,k1
        <= 18497526.8940961
 c1198: - aux_cap_b7,r8,band_UNPAIRED
        + 0.8035888618 aux_cap_helper__b7,r8,band_UNPAIRED__condX1
        + 1.1250244065 aux_cap_helper__b7,r8,band_UNPAIRED__condX2
        - 0.3214355447 aux_cap_helper__b7,r8,band_UNPAIRED__Z2  = 0
 c1199: X__b8,r3,band_UNPAIRED + X__b0,r3,band_UNPAIRED
        + X__b4,r3,band_UNPAIRED + X__b9,r3,band_UNPAIRED
        + X__b5,r3,band_UNPAIRED + X__b6,r3,band_UNPAIRED
        + X__b3,r3,band_UNPAIRED + X__b1,r3,band_UNPAIRED
        + X__b2,r3,band_UNPAIRED + X__b7,r3,band_UNPAIRED <= 2
 c1200: - aux_cap_b1,r1,band_HIGH_PAIRED
        - 0.6181448971 aux_cap_helper__b1,r1,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b1,r1,band_HIGH_PAIRED__Z3
        + 1.5453622427 aux_cap_helper__b1,r1,band_HIGH_PAIRED__condX1
        + 1.3908260184 aux_cap_helper__b1,r1,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b1,r1,band_HIGH_PAIRED__condX2  = 0
 c1201: 0.927217345653617 aux_cap_helper__b8,r7,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b8,r7,band_HIGH_PAIRED__Z2
        - aux_cap_b8,r7,band_HIGH_PAIRED
        + 1.3908260184 aux_cap_helper__b8,r7,band_HIGH_PAIRED__condX3
        + 1.5453622427 aux_cap_helper__b8,r7,band_HIGH_PAIRED__condX1
        + 2.1635071398 aux_cap_helper__b8,r7,band_HIGH_PAIRED__condX2  = 0
 c1202: 6.01974020941575 sv_function_helpervar_b3,r2__Z3
        - sv_function_helpervar_b3,r2__condX3 <= 0
 c1203: - aux_Omega_b3,r3 + 278.845760003569 aux_sv_b3,r3  = 0
 c1204: - aux_cap_helper__b2,r3,band_LOW_PAIRED__condX1 <= 0
 c1205: - aux_cap_b2,r10,band_LOW_PAIRED
        - 1.4943850124 aux_cap_helper__b2,r10,band_LOW_PAIRED__Z2
        + 5.2303475434 aux_cap_helper__b2,r10,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b2,r10,band_LOW_PAIRED__condX1  = 0
 c1206: - aux_cap_helper__b5,r4,band_LOW_PAIRED__Z1
        + aux_cap_helper__b5,r4,band_LOW_PAIRED__condX1 <= 0
 c1207: sv_function_helpervar_b4,r5__condX1
        - 0.15994618743266 sv_function_helpervar_b4,r5__Z1 <= 0
 c1208: aux_cap_helper__b3,r0,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b3,r0,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b3,r0,band_HIGH_PAIRED__Z3  = 1
 c1209: - 1.4943850124 aux_cap_helper__b5,r9,band_LOW_PAIRED__Z2
        + 5.2303475434 aux_cap_helper__b5,r9,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b5,r9,band_LOW_PAIRED__condX1
        - aux_cap_b5,r9,band_LOW_PAIRED  = 0
 c1210: - aux_sv_b0,r7 + 16.7702001874473 sv_function_helpervar_b0,r7__Z2
        + 36.6157076859395 sv_function_helpervar_b0,r7__Z3
        + 109.8706625607 sv_function_helpervar_b0,r7__condX1
        + 5.021647721 sv_function_helpervar_b0,r7__condX2
        + 1.7800321294 sv_function_helpervar_b0,r7__condX3  = 0
 c1211: sv_function_helpervar_b9,r2__Z2 + sv_function_helpervar_b9,r2__Z3
        + sv_function_helpervar_b9,r2__Z1  = 1
 c1212: - X__b3,r11,band_UNPAIRED
        + aux_cap_helper__b3,r11,band_UNPAIRED__condX1
        + aux_cap_helper__b3,r11,band_UNPAIRED__condX2  = 0
 c1213: aux_cap_helper__b2,r0,band_LOW_PAIRED__condX2
        + aux_cap_helper__b2,r0,band_LOW_PAIRED__condX1
        - X__b2,r0,band_LOW_PAIRED  = 0
 c1214: aux_cap_helper__b9,r5,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b9,r5,band_LOW_PAIRED__Z2 <= 0
 c1215: - 2 aux_cap_helper__b6,r1,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b6,r1,band_HIGH_PAIRED__condX2 <= 0
 c1216: - 2 aux_cap_helper__b4,r10,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b4,r10,band_HIGH_PAIRED__condX2 <= 0
 c1217: - sv_function_helpervar_b7,r3__condX2
        + 0.159946187436532 sv_function_helpervar_b7,r3__Z2 <= 0
 c1218: aux_cap_helper__b1,r1,band_UNPAIRED__Z1
        + aux_cap_helper__b1,r1,band_UNPAIRED__Z2  = 1
 c1219: aux_cap_helper__b2,r12,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b2,r12,band_UNPAIRED__Z2 <= 0
 c1220: aux_cap_helper__b1,r9,band_LOW_PAIRED__Z2
        + aux_cap_helper__b1,r9,band_LOW_PAIRED__Z1  = 1
 c1221: 0.927217345653617 aux_cap_helper__b4,r5,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b4,r5,band_HIGH_PAIRED__Z2
        - aux_cap_b4,r5,band_HIGH_PAIRED
        + 1.3908260184 aux_cap_helper__b4,r5,band_HIGH_PAIRED__condX3
        + 1.5453622427 aux_cap_helper__b4,r5,band_HIGH_PAIRED__condX1
        + 2.1635071398 aux_cap_helper__b4,r5,band_HIGH_PAIRED__condX2  = 0
 c1222: X__b7,r9,band_LOW_PAIRED - GlobalBidder_W_ir__b7,r9
        + X__b7,r9,band_UNPAIRED + X__b7,r9,band_HIGH_PAIRED >= 0
 c1223: 1.5453622427 aux_cap_helper__b3,r5,band_HIGH_PAIRED__condX1
        + 2.1635071398 aux_cap_helper__b3,r5,band_HIGH_PAIRED__condX2
        + 1.3908260184 aux_cap_helper__b3,r5,band_HIGH_PAIRED__condX3
        - 0.6181448971 aux_cap_helper__b3,r5,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b3,r5,band_HIGH_PAIRED__Z3
        - aux_cap_b3,r5,band_HIGH_PAIRED  = 0
 c1224: aux_cap_helper__b7,r7,band_UNPAIRED__condX2
        + aux_cap_helper__b7,r7,band_UNPAIRED__condX1 - X__b7,r7,band_UNPAIRED
         = 0
 c1225: - sv_function_helpervar_b0,r8__condX1 <= 0
 c1226: - aux_cap_b9,r10,band_UNPAIRED
        + 1.1250244065 aux_cap_helper__b9,r10,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b9,r10,band_UNPAIRED__condX1
        - 0.3214355447 aux_cap_helper__b9,r10,band_UNPAIRED__Z2  = 0
 c1227: - 2 aux_cap_helper__b8,r7,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b8,r7,band_HIGH_PAIRED__condX2 <= 0
 c1228: 1.3908260184 aux_cap_helper__b7,r6,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b7,r6,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b7,r6,band_HIGH_PAIRED__condX1
        - 0.6181448971 aux_cap_helper__b7,r6,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b7,r6,band_HIGH_PAIRED__Z3
        - aux_cap_b7,r6,band_HIGH_PAIRED  = 0
 c1229: aux_cap_helper__b1,r8,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b1,r8,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b1,r8,band_HIGH_PAIRED__Z3  = 1
 c1230: aux_cap_helper__b0,r10,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b0,r10,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b0,r10,band_HIGH_PAIRED__condX3
        - X__b0,r10,band_HIGH_PAIRED  = 0
 c1231: - aux_cap_b6,r6,band_HIGH_PAIRED
        + 1.3908260184 aux_cap_helper__b6,r6,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b6,r6,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b6,r6,band_HIGH_PAIRED__condX1
        + 0.927217345653617 aux_cap_helper__b6,r6,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b6,r6,band_HIGH_PAIRED__Z2  = 0
 c1232: 0.0102040816326531 X__b7,r1,band_HIGH_PAIRED
        + 0.0102040816326531 X__b7,r1,band_LOW_PAIRED
        - GlobalBidder_W_ir__b7,r1 + 0.0102040816326531 X__b7,r1,band_UNPAIRED
        <= 0
 c1233: aux_cap_helper__b3,r0,band_LOW_PAIRED__Z1
        + aux_cap_helper__b3,r0,band_LOW_PAIRED__Z2  = 1
 c1234: - 3 aux_cap_helper__b2,r8,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b2,r8,band_HIGH_PAIRED__condX3 <= 0
 c1235: - 2 aux_cap_helper__b7,r7,band_LOW_PAIRED__Z2
        + aux_cap_helper__b7,r7,band_LOW_PAIRED__condX2 <= 0
 c1236: - 0.159946187460821 sv_function_helpervar_b3,r4__Z1
        + sv_function_helpervar_b3,r4__condX1 <= 0
 c1237: aux_cap_helper__b5,r13,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b5,r13,band_LOW_PAIRED__Z2 <= 0
 c1238: 1.5453622427 aux_cap_helper__b9,r7,band_HIGH_PAIRED__condX1
        + 2.1635071398 aux_cap_helper__b9,r7,band_HIGH_PAIRED__condX2
        + 1.3908260184 aux_cap_helper__b9,r7,band_HIGH_PAIRED__condX3
        + 0.927217345653617 aux_cap_helper__b9,r7,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b9,r7,band_HIGH_PAIRED__Z2
        - aux_cap_b9,r7,band_HIGH_PAIRED  = 0
 c1239: - aux_cap_helper__b0,r4,band_LOW_PAIRED__Z1
        + aux_cap_helper__b0,r4,band_LOW_PAIRED__condX1 <= 0
 c1240: 1.5453622427 aux_cap_helper__b7,r12,band_HIGH_PAIRED__condX1
        + 1.3908260184 aux_cap_helper__b7,r12,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b7,r12,band_HIGH_PAIRED__condX2
        + 0.927217345653617 aux_cap_helper__b7,r12,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b7,r12,band_HIGH_PAIRED__Z2
        - aux_cap_b7,r12,band_HIGH_PAIRED  = 0
 c1241: - X__b6,r7,band_HIGH_PAIRED
        + aux_cap_helper__b6,r7,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b6,r7,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b6,r7,band_HIGH_PAIRED__condX3  = 0
 c1242: aux_cap_b8,r0,band_HIGH_PAIRED - aux_c_b8,r0
        + aux_cap_b8,r0,band_LOW_PAIRED + aux_cap_b8,r0,band_UNPAIRED  = 0
 c1243: - 1.4943850124 aux_cap_helper__b2,r12,band_LOW_PAIRED__Z2
        + 5.2303475434 aux_cap_helper__b2,r12,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b2,r12,band_LOW_PAIRED__condX1
        - aux_cap_b2,r12,band_LOW_PAIRED  = 0
 c1244: - X__b2,r13,band_LOW_PAIRED
        + aux_cap_helper__b2,r13,band_LOW_PAIRED__condX2
        + aux_cap_helper__b2,r13,band_LOW_PAIRED__condX1  = 0
 c1245: aux_cap_b4,r11,band_HIGH_PAIRED + aux_cap_b4,r11,band_LOW_PAIRED
        + aux_cap_b4,r11,band_UNPAIRED - aux_c_b4,r11  = 0
 c1246: 2.1635071398 aux_cap_helper__b1,r5,band_HIGH_PAIRED__condX2
        + 1.3908260184 aux_cap_helper__b1,r5,band_HIGH_PAIRED__condX3
        + 1.5453622427 aux_cap_helper__b1,r5,band_HIGH_PAIRED__condX1
        + 0.927217345653617 aux_cap_helper__b1,r5,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b1,r5,band_HIGH_PAIRED__Z2
        - aux_cap_b1,r5,band_HIGH_PAIRED  = 0
 c1247: - sv_function_helpervar_b3,r10__condX1 <= 0
 c1248: aux_cap_helper__b5,r6,band_UNPAIRED__Z1
        + aux_cap_helper__b5,r6,band_UNPAIRED__Z2  = 1
 c1249: aux_cap_helper__b8,r6,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b8,r6,band_UNPAIRED__Z2 <= 0
 c1250: - sv_function_helpervar_b4,r0__condX1 <= 0
 c1251: aux_Omega_b1,r2 + aux_Omega_b1,r5 + aux_Omega_b1,r0 - v__1
        + aux_Omega_b1,r10 + aux_Omega_b1,r11 + aux_Omega_b1,r12  = 0
 c1252: - aux_cap_helper__b2,r13,band_HIGH_PAIRED__condX1 <= 0
 c1253: - aux_cap_b5,r6,band_HIGH_PAIRED
        - 0.6181448971 aux_cap_helper__b5,r6,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b5,r6,band_HIGH_PAIRED__Z3
        + 1.3908260184 aux_cap_helper__b5,r6,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b5,r6,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b5,r6,band_HIGH_PAIRED__condX1  = 0
 c1254: 1616.46475187171 aux_sv_b0,r4 - aux_Omega_b0,r4  = 0
 c1255: aux_cap_helper__b7,r2,band_LOW_PAIRED__condX2
        + aux_cap_helper__b7,r2,band_LOW_PAIRED__condX1
        - X__b7,r2,band_LOW_PAIRED  = 0
 c1256: sv_function_helpervar_b3,r4__Z1 + sv_function_helpervar_b3,r4__Z3
        + sv_function_helpervar_b3,r4__Z2  = 1
 c1257: - 2 aux_cap_helper__b0,r10,band_LOW_PAIRED__Z2
        + aux_cap_helper__b0,r10,band_LOW_PAIRED__condX2 <= 0
 c1258: aux_cap_helper__b6,r9,band_UNPAIRED__Z2
        - aux_cap_helper__b6,r9,band_UNPAIRED__condX2 <= 0
 c1259: - 2 aux_cap_helper__b6,r12,band_LOW_PAIRED__Z2
        + aux_cap_helper__b6,r12,band_LOW_PAIRED__condX2 <= 0
 c1260: sv_function_helpervar_b8,r5__Z1 + sv_function_helpervar_b8,r5__Z3
        + sv_function_helpervar_b8,r5__Z2  = 1
 c1261: aux_Omega_b7,r10 + aux_Omega_b7,r11 + aux_Omega_b7,r12
        + aux_Omega_b7,r13 + aux_Omega_b7,r0 + aux_Omega_b7,r1
        + aux_Omega_b7,r2 + aux_Omega_b7,r3 + aux_Omega_b7,r4 + aux_Omega_b7,r5
        + aux_Omega_b7,r6 + aux_Omega_b7,r7 + aux_Omega_b7,r8 + aux_Omega_b7,r9
        - GlobalBidder_PSI__i7,k1 - 18497526.8940961 GlobalBidder_hat{W}__i7,k1
        >= -18497526.8940961
 c1262: - aux_cap_helper__b5,r4,band_UNPAIRED__condX2
        + aux_cap_helper__b5,r4,band_UNPAIRED__Z2 <= 0
 c1263: aux_cap_helper__b9,r13,band_LOW_PAIRED__Z2
        - aux_cap_helper__b9,r13,band_LOW_PAIRED__condX2 <= 0
 c1264: - 0.159946187454412 sv_function_helpervar_b0,r0__Z1
        + sv_function_helpervar_b0,r0__condX1 <= 0
 c1265: aux_cap_helper__b8,r12,band_LOW_PAIRED__condX2
        + aux_cap_helper__b8,r12,band_LOW_PAIRED__condX1
        - X__b8,r12,band_LOW_PAIRED  = 0
 c1266: - aux_cap_helper__b7,r11,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b7,r11,band_HIGH_PAIRED__Z3 <= 0
 c1267: - sv_function_helpervar_b8,r7__condX2
        + 0.159946187436067 sv_function_helpervar_b8,r7__Z2 <= 0
 c1268: - aux_cap_helper__b6,r9,band_UNPAIRED__condX1 <= 0
 c1269: sv_function_helpervar_b4,r9__condX3
        - 15.9946187432626 sv_function_helpervar_b4,r9__Z3 <= 0
 c1270: - aux_cap_b7,r6,band_LOW_PAIRED
        - 1.4943850124 aux_cap_helper__b7,r6,band_LOW_PAIRED__Z2
        + 3.735962531 aux_cap_helper__b7,r6,band_LOW_PAIRED__condX1
        + 5.2303475434 aux_cap_helper__b7,r6,band_LOW_PAIRED__condX2  = 0
 c1271: aux_cap_helper__b6,r13,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b6,r13,band_HIGH_PAIRED__Z3 <= 0
 c1272: aux_cap_b4,r8,band_LOW_PAIRED + aux_cap_b4,r8,band_UNPAIRED
        - aux_c_b4,r8 + aux_cap_b4,r8,band_HIGH_PAIRED  = 0
 c1273: aux_cap_helper__b0,r11,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b0,r11,band_HIGH_PAIRED__Z3 <= 0
 c1274: X__b8,r5,band_HIGH_PAIRED + X__b1,r5,band_HIGH_PAIRED
        + X__b4,r5,band_HIGH_PAIRED + X__b9,r5,band_HIGH_PAIRED
        + X__b5,r5,band_HIGH_PAIRED + X__b7,r5,band_HIGH_PAIRED
        + X__b6,r5,band_HIGH_PAIRED + X__b2,r5,band_HIGH_PAIRED
        + X__b3,r5,band_HIGH_PAIRED + X__b0,r5,band_HIGH_PAIRED <= 3
 c1275: - 5.85581773625452 sv_function_helpervar_b2,r7__Z2
        + sv_function_helpervar_b2,r7__condX2 <= 0
 c1276: 0.01314 aux_Omega_b4,r0 + 0.17678 aux_Omega_b4,r9
        + 0.07433 aux_Omega_b4,r8 + 0.03125 aux_Omega_b4,r7
        + 0.03125 aux_Omega_b4,r6 + 0.03125 aux_Omega_b4,r5
        + 0.01314 aux_Omega_b4,r4 + 0.01314 aux_Omega_b4,r3
        + 0.00552 aux_Omega_b4,r2 + 0.00552 aux_Omega_b4,r1 - v__4
        + 0.42045 aux_Omega_b4,r12 + 0.42045 aux_Omega_b4,r13
        + 0.42045 aux_Omega_b4,r10 + aux_Omega_b4,r11  = 0
 c1277: - aux_cap_helper__b5,r4,band_UNPAIRED__condX1 <= 0
 c1278: - aux_cap_helper__b9,r13,band_LOW_PAIRED__condX1 <= 0
 c1279: 210.683651630829 sv_function_helpervar_b3,r11__Z2
        + 457.389766827553 sv_function_helpervar_b3,r11__Z3
        + 1379.5885613423 sv_function_helpervar_b3,r11__condX1
        + 62.3727217875 sv_function_helpervar_b3,r11__condX2
        + 22.4993944202 sv_function_helpervar_b3,r11__condX3 - aux_sv_b3,r11
         = 0
 c1280: - aux_cap_helper__b2,r2,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b2,r2,band_HIGH_PAIRED__Z3 <= 0
 c1281: aux_cap_helper__b4,r2,band_UNPAIRED__Z2
        + aux_cap_helper__b4,r2,band_UNPAIRED__Z1  = 1
 c1282: 0.927217345653617 aux_cap_helper__b8,r6,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b8,r6,band_HIGH_PAIRED__Z2
        - aux_cap_b8,r6,band_HIGH_PAIRED
        + 1.5453622427 aux_cap_helper__b8,r6,band_HIGH_PAIRED__condX1
        + 2.1635071398 aux_cap_helper__b8,r6,band_HIGH_PAIRED__condX2
        + 1.3908260184 aux_cap_helper__b8,r6,band_HIGH_PAIRED__condX3  = 0
 c1283: - 1.4943850124 aux_cap_helper__b2,r8,band_LOW_PAIRED__Z2
        - aux_cap_b2,r8,band_LOW_PAIRED
        + 3.735962531 aux_cap_helper__b2,r8,band_LOW_PAIRED__condX1
        + 5.2303475434 aux_cap_helper__b2,r8,band_LOW_PAIRED__condX2  = 0
 c1284: - aux_cap_helper__b4,r8,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b4,r8,band_HIGH_PAIRED__Z3 <= 0
 c1285: sv_function_helpervar_b0,r3__Z3 + sv_function_helpervar_b0,r3__Z2
        + sv_function_helpervar_b0,r3__Z1  = 1
 c1286: sv_function_helpervar_b1,r9__condX3
        - 15.9946187432626 sv_function_helpervar_b1,r9__Z3 <= 0
 c1287: - X__b7,r9,band_LOW_PAIRED
        + aux_cap_helper__b7,r9,band_LOW_PAIRED__condX2
        + aux_cap_helper__b7,r9,band_LOW_PAIRED__condX1  = 0
 c1288: aux_cap_helper__b4,r4,band_LOW_PAIRED__Z1
        + aux_cap_helper__b4,r4,band_LOW_PAIRED__Z2  = 1
 c1289: aux_cap_helper__b8,r3,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b8,r3,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b8,r3,band_HIGH_PAIRED__Z2  = 1
 c1290: aux_cap_helper__b8,r2,band_LOW_PAIRED__Z2
        + aux_cap_helper__b8,r2,band_LOW_PAIRED__Z1  = 1
 c1291: - 2 aux_cap_helper__b0,r9,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b0,r9,band_HIGH_PAIRED__condX2 <= 0
 c1292: aux_cap_helper__b7,r7,band_LOW_PAIRED__Z1
        + aux_cap_helper__b7,r7,band_LOW_PAIRED__Z2  = 1
 c1293: sv_function_helpervar_b3,r8__condX3
        + sv_function_helpervar_b3,r8__condX2
        + sv_function_helpervar_b3,r8__condX1 - aux_c_b3,r8  = 0
 c1294: aux_cap_helper__b6,r0,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b6,r0,band_HIGH_PAIRED__Z1 <= 0
 c1295: - aux_cap_helper__b2,r1,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b2,r1,band_HIGH_PAIRED__Z3 <= 0
 c1296: - aux_cap_b2,r6,band_HIGH_PAIRED
        + 1.3908260184 aux_cap_helper__b2,r6,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b2,r6,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b2,r6,band_HIGH_PAIRED__condX1
        + 0.927217345653617 aux_cap_helper__b2,r6,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b2,r6,band_HIGH_PAIRED__Z2  = 0
 c1297: - aux_c_b5,r5 + sv_function_helpervar_b5,r5__condX3
        + sv_function_helpervar_b5,r5__condX2
        + sv_function_helpervar_b5,r5__condX1  = 0
 c1298: aux_cap_helper__b9,r6,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b9,r6,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b9,r6,band_HIGH_PAIRED__Z2  = 1
 c1299: - 15.9946187432626 sv_function_helpervar_b2,r7__Z3
        + sv_function_helpervar_b2,r7__condX3 <= 0
 c1300: aux_cap_helper__b9,r7,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b9,r7,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b9,r7,band_HIGH_PAIRED__condX3
        - X__b9,r7,band_HIGH_PAIRED  = 0
 c1301: sv_function_helpervar_b2,r11__Z3 + sv_function_helpervar_b2,r11__Z2
        + sv_function_helpervar_b2,r11__Z1  = 1
 c1302: 0.15994618748582 sv_function_helpervar_b2,r7__Z2
        - sv_function_helpervar_b2,r7__condX2 <= 0
 c1303: aux_cap_helper__b3,r7,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b3,r7,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b3,r7,band_HIGH_PAIRED__condX3
        - X__b3,r7,band_HIGH_PAIRED  = 0
 c1304: - 2 aux_cap_helper__b8,r8,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b8,r8,band_HIGH_PAIRED__condX2 <= 0
 c1305: aux_cap_helper__b2,r0,band_UNPAIRED__condX2
        + aux_cap_helper__b2,r0,band_UNPAIRED__condX1 - X__b2,r0,band_UNPAIRED
         = 0
 c1306: sv_function_helpervar_b9,r9__Z3 + sv_function_helpervar_b9,r9__Z2
        + sv_function_helpervar_b9,r9__Z1  = 1
 c1307: aux_cap_helper__b2,r2,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b2,r2,band_UNPAIRED__Z2 <= 0
 c1308: 21.399513295 sv_function_helpervar_b3,r3__condX3
        + 68.0651342123 sv_function_helpervar_b3,r3__condX2
        + 1379.5885610589 sv_function_helpervar_b3,r3__condX1 - aux_sv_b3,r3
        + 209.773171963657 sv_function_helpervar_b3,r3__Z2
        + 474.981946087648 sv_function_helpervar_b3,r3__Z3  = 0
 c1309: aux_cap_helper__b1,r13,band_UNPAIRED__condX1
        - aux_cap_helper__b1,r13,band_UNPAIRED__Z1 <= 0
 c1310: aux_cap_helper__b3,r8,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b3,r8,band_LOW_PAIRED__Z2 <= 0
 c1311: aux_cap_helper__b0,r3,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b0,r3,band_HIGH_PAIRED__Z3 <= 0
 c1312: - 0.6181448971 aux_cap_helper__b3,r6,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b3,r6,band_HIGH_PAIRED__Z3
        - aux_cap_b3,r6,band_HIGH_PAIRED
        + 2.1635071398 aux_cap_helper__b3,r6,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b3,r6,band_HIGH_PAIRED__condX1
        + 1.3908260184 aux_cap_helper__b3,r6,band_HIGH_PAIRED__condX3  = 0
 c1313: 3.735962531 aux_cap_helper__b6,r9,band_LOW_PAIRED__condX1
        + 5.2303475434 aux_cap_helper__b6,r9,band_LOW_PAIRED__condX2
        - aux_cap_b6,r9,band_LOW_PAIRED
        - 1.4943850124 aux_cap_helper__b6,r9,band_LOW_PAIRED__Z2  = 0
 c1314: aux_cap_helper__b7,r5,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b7,r5,band_HIGH_PAIRED__condX2 <= 0
 c1315: - X__b9,r5,band_UNPAIRED + aux_cap_helper__b9,r5,band_UNPAIRED__condX2
        + aux_cap_helper__b9,r5,band_UNPAIRED__condX1  = 0
 c1316: - 2 aux_cap_helper__b4,r7,band_UNPAIRED__Z2
        + aux_cap_helper__b4,r7,band_UNPAIRED__condX2 <= 0
 c1317: aux_cap_helper__b8,r11,band_LOW_PAIRED__Z2
        + aux_cap_helper__b8,r11,band_LOW_PAIRED__Z1  = 1
 c1318: 366.353954418981 sv_function_helpervar_b6,r5__Z3
        + 161.002044561677 sv_function_helpervar_b6,r5__Z2
        + 1059.4718672271 sv_function_helpervar_b6,r5__condX1
        + 16.3348731016 sv_function_helpervar_b6,r5__condX3
        + 52.8705402958 sv_function_helpervar_b6,r5__condX2 - aux_sv_b6,r5  = 0
 c1319: aux_cap_helper__b8,r1,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b8,r1,band_UNPAIRED__Z2 <= 0
 c1320: sv_function_helpervar_b4,r8__condX2
        - 6.01793528589529 sv_function_helpervar_b4,r8__Z2 <= 0
 c1321: - 3 aux_cap_helper__b5,r5,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b5,r5,band_HIGH_PAIRED__condX3 <= 0
 c1322: aux_cap_helper__b4,r3,band_LOW_PAIRED__condX1
        - aux_cap_helper__b4,r3,band_LOW_PAIRED__Z1 <= 0
 c1323: aux_cap_helper__b5,r9,band_UNPAIRED__condX2
        + aux_cap_helper__b5,r9,band_UNPAIRED__condX1 - X__b5,r9,band_UNPAIRED
         = 0
 c1324: - aux_cap_b0,r2,band_HIGH_PAIRED
        + 2.1635071398 aux_cap_helper__b0,r2,band_HIGH_PAIRED__condX2
        + 1.3908260184 aux_cap_helper__b0,r2,band_HIGH_PAIRED__condX3
        + 1.5453622427 aux_cap_helper__b0,r2,band_HIGH_PAIRED__condX1
        - 0.6181448971 aux_cap_helper__b0,r2,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b0,r2,band_HIGH_PAIRED__Z3  = 0
 c1325: - 1.4943850124 aux_cap_helper__b3,r5,band_LOW_PAIRED__Z2
        - aux_cap_b3,r5,band_LOW_PAIRED
        + 5.2303475434 aux_cap_helper__b3,r5,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b3,r5,band_LOW_PAIRED__condX1  = 0
 c1326: - 15.9946187432626 sv_function_helpervar_b5,r7__Z3
        + sv_function_helpervar_b5,r7__condX3 <= 0
 c1327: - aux_cap_helper__b7,r5,band_HIGH_PAIRED__condX1 <= 0
 c1328: 1.3908260184 aux_cap_helper__b4,r3,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b4,r3,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b4,r3,band_HIGH_PAIRED__condX1
        - aux_cap_b4,r3,band_HIGH_PAIRED
        + 0.927217345653617 aux_cap_helper__b4,r3,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b4,r3,band_HIGH_PAIRED__Z2  = 0
 c1329: sv_function_helpervar_b9,r12__condX2
        + sv_function_helpervar_b9,r12__condX1
        + sv_function_helpervar_b9,r12__condX3 - aux_c_b9,r12  = 0
 c1330: aux_cap_helper__b7,r1,band_UNPAIRED__condX1
        - aux_cap_helper__b7,r1,band_UNPAIRED__Z1 <= 0
 c1331: sv_function_helpervar_b4,r5__condX2
        + sv_function_helpervar_b4,r5__condX3
        + sv_function_helpervar_b4,r5__condX1 - aux_c_b4,r5  = 0
 c1332: - X__b4,r7,band_LOW_PAIRED
        + aux_cap_helper__b4,r7,band_LOW_PAIRED__condX2
        + aux_cap_helper__b4,r7,band_LOW_PAIRED__condX1  = 0
 c1333: - aux_cap_helper__b7,r10,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b7,r10,band_HIGH_PAIRED__Z3 <= 0
 c1334: - aux_cap_helper__b0,r5,band_UNPAIRED__condX1 <= 0
 c1335: - aux_cap_helper__b5,r9,band_UNPAIRED__condX1 <= 0
 c1336: 53.5919450309191 sv_function_helpervar_b2,r12__Z3
        + 24.5498664129444 sv_function_helpervar_b2,r12__Z2 - aux_sv_b2,r12
        + 7.3484818475 sv_function_helpervar_b2,r12__condX2
        + 2.6062939113 sv_function_helpervar_b2,r12__condX3
        + 160.8367693938 sv_function_helpervar_b2,r12__condX1  = 0
 c1337: aux_cap_helper__b6,r10,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b6,r10,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b6,r10,band_HIGH_PAIRED__Z2  = 1
 c1338: sv_function_helpervar_b1,r9__condX2
        - 6.6286062074629 sv_function_helpervar_b1,r9__Z2 <= 0
 c1339: - aux_cap_b9,r6,band_HIGH_PAIRED
        + 1.3908260184 aux_cap_helper__b9,r6,band_HIGH_PAIRED__condX3
        + 1.5453622427 aux_cap_helper__b9,r6,band_HIGH_PAIRED__condX1
        + 2.1635071398 aux_cap_helper__b9,r6,band_HIGH_PAIRED__condX2
        + 0.927217345653617 aux_cap_helper__b9,r6,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b9,r6,band_HIGH_PAIRED__Z2  = 0
 c1340: aux_cap_helper__b7,r0,band_LOW_PAIRED__Z2
        - aux_cap_helper__b7,r0,band_LOW_PAIRED__condX2 <= 0
 c1341: 1.1250244065 aux_cap_helper__b6,r11,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b6,r11,band_UNPAIRED__condX1
        - 0.3214355447 aux_cap_helper__b6,r11,band_UNPAIRED__Z2
        - aux_cap_b6,r11,band_UNPAIRED  = 0
 c1342: aux_cap_helper__b7,r6,band_UNPAIRED__condX1
        - aux_cap_helper__b7,r6,band_UNPAIRED__Z1 <= 0
 c1343: X__b8,r11,band_LOW_PAIRED + X__b8,r11,band_HIGH_PAIRED
        - GlobalBidder_W_ir__b8,r11 + X__b8,r11,band_UNPAIRED >= 0
 c1344: aux_cap_helper__b2,r4,band_LOW_PAIRED__Z2
        + aux_cap_helper__b2,r4,band_LOW_PAIRED__Z1  = 1
 c1345: aux_cap_helper__b2,r2,band_LOW_PAIRED__condX1
        - aux_cap_helper__b2,r2,band_LOW_PAIRED__Z1 <= 0
 c1346: aux_cap_helper__b4,r1,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b4,r1,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b4,r1,band_HIGH_PAIRED__Z1  = 1
 c1347: - X__b4,r6,band_HIGH_PAIRED
        + aux_cap_helper__b4,r6,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b4,r6,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b4,r6,band_HIGH_PAIRED__condX2  = 0
 c1348: - aux_cap_helper__b7,r0,band_LOW_PAIRED__condX1 <= 0
 c1349: - aux_cap_b5,r2,band_UNPAIRED
        + 0.8035888618 aux_cap_helper__b5,r2,band_UNPAIRED__condX1
        + 1.1250244065 aux_cap_helper__b5,r2,band_UNPAIRED__condX2
        - 0.3214355447 aux_cap_helper__b5,r2,band_UNPAIRED__Z2  = 0
 c1350: - aux_cap_b5,r5,band_HIGH_PAIRED
        + 0.927217345653617 aux_cap_helper__b5,r5,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b5,r5,band_HIGH_PAIRED__Z2
        + 1.5453622427 aux_cap_helper__b5,r5,band_HIGH_PAIRED__condX1
        + 1.3908260184 aux_cap_helper__b5,r5,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b5,r5,band_HIGH_PAIRED__condX2  = 0
 c1351: aux_cap_helper__b0,r5,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b0,r5,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b0,r5,band_HIGH_PAIRED__Z3  = 1
 c1352: 16.7024728683806 sv_function_helpervar_b0,r10__Z2
        + 37.8928258972495 sv_function_helpervar_b0,r10__Z3
        + 109.8706626074 sv_function_helpervar_b0,r10__condX1
        + 1.7001853865 sv_function_helpervar_b0,r10__condX3
        + 5.4450858816 sv_function_helpervar_b0,r10__condX2 - aux_sv_b0,r10
         = 0
 c1353: aux_cap_helper__b7,r4,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b7,r4,band_LOW_PAIRED__Z2 <= 0
 c1354: 0.159946187486662 sv_function_helpervar_b6,r5__Z2
        - sv_function_helpervar_b6,r5__condX2 <= 0
 c1355: 0.159946187451481 sv_function_helpervar_b4,r0__Z2
        - sv_function_helpervar_b4,r0__condX2 <= 0
 c1356: aux_cap_helper__b0,r5,band_UNPAIRED__Z2
        - aux_cap_helper__b0,r5,band_UNPAIRED__condX2 <= 0
 c1357: - 0.6181448971 aux_cap_helper__b2,r13,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b2,r13,band_HIGH_PAIRED__Z3
        - aux_cap_b2,r13,band_HIGH_PAIRED
        + 1.3908260184 aux_cap_helper__b2,r13,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b2,r13,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b2,r13,band_HIGH_PAIRED__condX1  = 0
 c1358: aux_cap_helper__b8,r7,band_UNPAIRED__Z1
        + aux_cap_helper__b8,r7,band_UNPAIRED__Z2  = 1
 c1359: - 2 aux_cap_helper__b5,r9,band_LOW_PAIRED__Z2
        + aux_cap_helper__b5,r9,band_LOW_PAIRED__condX2 <= 0
 c1360: - aux_cap_b2,r5,band_HIGH_PAIRED
        + 1.5453622427 aux_cap_helper__b2,r5,band_HIGH_PAIRED__condX1
        + 2.1635071398 aux_cap_helper__b2,r5,band_HIGH_PAIRED__condX2
        + 1.3908260184 aux_cap_helper__b2,r5,band_HIGH_PAIRED__condX3
        + 0.927217345653617 aux_cap_helper__b2,r5,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b2,r5,band_HIGH_PAIRED__Z2  = 0
 c1361: - aux_cap_b0,r9,band_UNPAIRED
        - 0.3214355447 aux_cap_helper__b0,r9,band_UNPAIRED__Z2
        + 0.8035888618 aux_cap_helper__b0,r9,band_UNPAIRED__condX1
        + 1.1250244065 aux_cap_helper__b0,r9,band_UNPAIRED__condX2  = 0
 c1362: - aux_cap_helper__b5,r9,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b5,r9,band_HIGH_PAIRED__condX1 <= 0
 c1363: - 15.9946187432626 sv_function_helpervar_b7,r10__Z3
        + sv_function_helpervar_b7,r10__condX3 <= 0
 c1364: 0.0102040816326531 X__b8,r5,band_HIGH_PAIRED
        + 0.0102040816326531 X__b8,r5,band_LOW_PAIRED
        + 0.0102040816326531 X__b8,r5,band_UNPAIRED - GlobalBidder_W_ir__b8,r5
        <= 0
 c1365: - aux_cap_helper__b3,r3,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b3,r3,band_HIGH_PAIRED__condX1 <= 0
 c1366: - aux_cap_helper__b2,r3,band_HIGH_PAIRED__condX1 <= 0
 c1367: aux_cap_helper__b8,r2,band_UNPAIRED__condX2
        + aux_cap_helper__b8,r2,band_UNPAIRED__condX1 - X__b8,r2,band_UNPAIRED
         = 0
 c1368: aux_cap_helper__b8,r9,band_UNPAIRED__Z2
        - aux_cap_helper__b8,r9,band_UNPAIRED__condX2 <= 0
 c1369: sv_function_helpervar_b4,r2__condX1
        + sv_function_helpervar_b4,r2__condX2
        + sv_function_helpervar_b4,r2__condX3 - aux_c_b4,r2  = 0
 c1370: - 3 aux_cap_helper__b5,r6,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b5,r6,band_HIGH_PAIRED__condX3 <= 0
 c1371: aux_cap_helper__b0,r11,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b0,r11,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b0,r11,band_HIGH_PAIRED__Z3  = 1
 c1372: - aux_cap_b5,r3,band_HIGH_PAIRED
        + 0.927217345653617 aux_cap_helper__b5,r3,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b5,r3,band_HIGH_PAIRED__Z2
        + 1.5453622427 aux_cap_helper__b5,r3,band_HIGH_PAIRED__condX1
        + 1.3908260184 aux_cap_helper__b5,r3,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b5,r3,band_HIGH_PAIRED__condX2  = 0
 c1373: 0.927217345653617 aux_cap_helper__b0,r1,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b0,r1,band_HIGH_PAIRED__Z2
        + 2.1635071398 aux_cap_helper__b0,r1,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b0,r1,band_HIGH_PAIRED__condX1
        + 1.3908260184 aux_cap_helper__b0,r1,band_HIGH_PAIRED__condX3
        - aux_cap_b0,r1,band_HIGH_PAIRED  = 0
 c1374: - 6.34405022357509 sv_function_helpervar_b0,r13__Z2
        + sv_function_helpervar_b0,r13__condX2 <= 0
 c1375: - aux_cap_helper__b9,r9,band_LOW_PAIRED__condX1 <= 0
 c1376: X__b9,r12,band_LOW_PAIRED + X__b2,r12,band_LOW_PAIRED
        + X__b1,r12,band_LOW_PAIRED + X__b0,r12,band_LOW_PAIRED
        + X__b4,r12,band_LOW_PAIRED + X__b3,r12,band_LOW_PAIRED
        + X__b8,r12,band_LOW_PAIRED + X__b5,r12,band_LOW_PAIRED
        + X__b6,r12,band_LOW_PAIRED + X__b7,r12,band_LOW_PAIRED <= 2
 c1377: - aux_cap_helper__b8,r9,band_UNPAIRED__condX1 <= 0
 c1378: aux_cap_helper__b1,r6,band_UNPAIRED__condX2
        + aux_cap_helper__b1,r6,band_UNPAIRED__condX1 - X__b1,r6,band_UNPAIRED
         = 0
 c1379: - aux_cap_helper__b1,r11,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b1,r11,band_HIGH_PAIRED__Z3 <= 0
 c1380: 590.477080314056 aux_sv_b0,r3 - aux_Omega_b0,r3  = 0
 c1381: aux_cap_helper__b0,r12,band_UNPAIRED__Z2
        - aux_cap_helper__b0,r12,band_UNPAIRED__condX2 <= 0
 c1382: - 0.3214355447 aux_cap_helper__b5,r0,band_UNPAIRED__Z2
        - aux_cap_b5,r0,band_UNPAIRED
        + 1.1250244065 aux_cap_helper__b5,r0,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b5,r0,band_UNPAIRED__condX1  = 0
 c1383: aux_cap_helper__b9,r1,band_LOW_PAIRED__Z2
        - aux_cap_helper__b9,r1,band_LOW_PAIRED__condX2 <= 0
 c1384: GlobalBidder_W_i_7 + 13 GlobalBidder_hat{W}__i7,k2 >= 13
 c1385: - sv_function_helpervar_b1,r4__condX3
        + 6.26387748824355 sv_function_helpervar_b1,r4__Z3 <= 0
 c1386: - aux_cap_helper__b9,r10,band_LOW_PAIRED__condX2
        + aux_cap_helper__b9,r10,band_LOW_PAIRED__Z2 <= 0
 c1387: 1.5453622427 aux_cap_helper__b3,r2,band_HIGH_PAIRED__condX1
        + 2.1635071398 aux_cap_helper__b3,r2,band_HIGH_PAIRED__condX2
        + 1.3908260184 aux_cap_helper__b3,r2,band_HIGH_PAIRED__condX3
        - 0.6181448971 aux_cap_helper__b3,r2,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b3,r2,band_HIGH_PAIRED__Z3
        - aux_cap_b3,r2,band_HIGH_PAIRED  = 0
 c1388: - X__b5,r11,band_HIGH_PAIRED
        + aux_cap_helper__b5,r11,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b5,r11,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b5,r11,band_HIGH_PAIRED__condX1  = 0
 c1389: - aux_cap_helper__b8,r11,band_LOW_PAIRED__condX1 <= 0
 c1390: 31.3211828106423 aux_sv_b5,r13 - aux_Omega_b5,r13  = 0
 c1391: - aux_cap_helper__b0,r12,band_UNPAIRED__condX1 <= 0
 c1392: aux_cap_helper__b4,r10,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b4,r10,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b4,r10,band_HIGH_PAIRED__condX1
        - X__b4,r10,band_HIGH_PAIRED  = 0
 c1393: - aux_cap_helper__b5,r9,band_UNPAIRED__condX2
        + aux_cap_helper__b5,r9,band_UNPAIRED__Z2 <= 0
 c1394: - sv_function_helpervar_b3,r9__condX3
        + 5.77656638381451 sv_function_helpervar_b3,r9__Z3 <= 0
 c1395: sv_function_helpervar_b8,r0__Z3 + sv_function_helpervar_b8,r0__Z2
        + sv_function_helpervar_b8,r0__Z1  = 1
 c1396: - aux_cap_helper__b9,r10,band_LOW_PAIRED__condX1 <= 0
 c1397: - 2 aux_cap_helper__b6,r13,band_UNPAIRED__Z2
        + aux_cap_helper__b6,r13,band_UNPAIRED__condX2 <= 0
 c1398: sv_function_helpervar_b2,r3__Z1 + sv_function_helpervar_b2,r3__Z2
        + sv_function_helpervar_b2,r3__Z3  = 1
 c1399: X__b5,r6,band_HIGH_PAIRED + X__b1,r6,band_HIGH_PAIRED
        + X__b8,r6,band_HIGH_PAIRED + X__b2,r6,band_HIGH_PAIRED
        + X__b4,r6,band_HIGH_PAIRED + X__b3,r6,band_HIGH_PAIRED
        + X__b7,r6,band_HIGH_PAIRED + X__b6,r6,band_HIGH_PAIRED
        + X__b0,r6,band_HIGH_PAIRED + X__b9,r6,band_HIGH_PAIRED <= 3
 c1400: aux_cap_helper__b7,r10,band_LOW_PAIRED__condX2
        + aux_cap_helper__b7,r10,band_LOW_PAIRED__condX1
        - X__b7,r10,band_LOW_PAIRED  = 0
 c1401: - aux_cap_helper__b2,r3,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b2,r3,band_HIGH_PAIRED__Z2 <= 0
 c1402: - aux_cap_b6,r7,band_UNPAIRED
        - 0.3214355447 aux_cap_helper__b6,r7,band_UNPAIRED__Z2
        + 1.1250244065 aux_cap_helper__b6,r7,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b6,r7,band_UNPAIRED__condX1  = 0
 c1403: 1098.95619437048 aux_sv_b8,r3 - aux_Omega_b8,r3  = 0
 c1404: aux_cap_helper__b9,r11,band_UNPAIRED__Z2
        + aux_cap_helper__b9,r11,band_UNPAIRED__Z1  = 1
 c1405: 0.8035888618 aux_cap_helper__b3,r5,band_UNPAIRED__condX1
        + 1.1250244065 aux_cap_helper__b3,r5,band_UNPAIRED__condX2
        - 0.3214355447 aux_cap_helper__b3,r5,band_UNPAIRED__Z2
        - aux_cap_b3,r5,band_UNPAIRED  = 0
 c1406: 5.2303475434 aux_cap_helper__b0,r7,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b0,r7,band_LOW_PAIRED__condX1
        - aux_cap_b0,r7,band_LOW_PAIRED
        - 1.4943850124 aux_cap_helper__b0,r7,band_LOW_PAIRED__Z2  = 0
 c1407: 6.34442616466423 sv_function_helpervar_b1,r3__Z3
        - sv_function_helpervar_b1,r3__condX3 <= 0
 c1408: - aux_cap_helper__b7,r12,band_HIGH_PAIRED__condX1 <= 0
 c1409: 1.3908260184 aux_cap_helper__b8,r3,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b8,r3,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b8,r3,band_HIGH_PAIRED__condX1
        + 0.927217345653617 aux_cap_helper__b8,r3,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b8,r3,band_HIGH_PAIRED__Z2
        - aux_cap_b8,r3,band_HIGH_PAIRED  = 0
 c1410: - X__b2,r9,band_UNPAIRED + aux_cap_helper__b2,r9,band_UNPAIRED__condX2
        + aux_cap_helper__b2,r9,band_UNPAIRED__condX1  = 0
 c1411: - aux_Omega_b6,r8 + 148.60867704181 aux_sv_b6,r8  = 0
 c1412: sv_function_helpervar_b4,r5__condX2
        - 6.37510953535038 sv_function_helpervar_b4,r5__Z2 <= 0
 c1413: - aux_sv_b6,r3 + 48.8842811111 sv_function_helpervar_b6,r3__condX2
        + 17.0674637544 sv_function_helpervar_b6,r3__condX3
        + 1059.4718673393 sv_function_helpervar_b6,r3__condX1
        + 354.636446232165 sv_function_helpervar_b6,r3__Z3
        + 161.639631521444 sv_function_helpervar_b6,r3__Z2  = 0
 c1414: - aux_cap_helper__b9,r9,band_LOW_PAIRED__condX2
        + aux_cap_helper__b9,r9,band_LOW_PAIRED__Z2 <= 0
 c1415: aux_cap_helper__b0,r1,band_LOW_PAIRED__condX1
        - aux_cap_helper__b0,r1,band_LOW_PAIRED__Z1 <= 0
 c1416: aux_cap_helper__b5,r0,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b5,r0,band_HIGH_PAIRED__condX2 <= 0
 c1417: aux_cap_helper__b4,r11,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b4,r11,band_HIGH_PAIRED__Z1 <= 0
 c1418: aux_cap_helper__b1,r0,band_LOW_PAIRED__Z2
        + aux_cap_helper__b1,r0,band_LOW_PAIRED__Z1  = 1
 c1419: - sv_function_helpervar_b2,r0__condX2
        + 0.159946187434522 sv_function_helpervar_b2,r0__Z2 <= 0
 c1420: - 0.159946187459929 sv_function_helpervar_b7,r4__Z1
        + sv_function_helpervar_b7,r4__condX1 <= 0
 c1421: - aux_cap_helper__b5,r0,band_HIGH_PAIRED__condX1 <= 0
 c1422: aux_cap_helper__b2,r0,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b2,r0,band_HIGH_PAIRED__condX2 <= 0
 c1423: - aux_cap_b4,r2,band_HIGH_PAIRED
        + 0.927217345653617 aux_cap_helper__b4,r2,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b4,r2,band_HIGH_PAIRED__Z2
        + 2.1635071398 aux_cap_helper__b4,r2,band_HIGH_PAIRED__condX2
        + 1.3908260184 aux_cap_helper__b4,r2,band_HIGH_PAIRED__condX3
        + 1.5453622427 aux_cap_helper__b4,r2,band_HIGH_PAIRED__condX1  = 0
 c1424: sv_function_helpervar_b5,r8__condX1
        + sv_function_helpervar_b5,r8__condX3
        + sv_function_helpervar_b5,r8__condX2 - aux_c_b5,r8  = 0
 c1425: - aux_cap_b3,r12,band_LOW_PAIRED
        + 3.735962531 aux_cap_helper__b3,r12,band_LOW_PAIRED__condX1
        + 5.2303475434 aux_cap_helper__b3,r12,band_LOW_PAIRED__condX2
        - 1.4943850124 aux_cap_helper__b3,r12,band_LOW_PAIRED__Z2  = 0
 c1426: - 2 aux_cap_helper__b9,r8,band_LOW_PAIRED__Z2
        + aux_cap_helper__b9,r8,band_LOW_PAIRED__condX2 <= 0
 c1427: aux_cap_helper__b0,r12,band_LOW_PAIRED__Z2
        + aux_cap_helper__b0,r12,band_LOW_PAIRED__Z1  = 1
 c1428: - aux_cap_helper__b2,r0,band_HIGH_PAIRED__condX1 <= 0
 c1429: - aux_cap_b1,r2,band_HIGH_PAIRED
        + 1.3908260184 aux_cap_helper__b1,r2,band_HIGH_PAIRED__condX3
        + 1.5453622427 aux_cap_helper__b1,r2,band_HIGH_PAIRED__condX1
        + 2.1635071398 aux_cap_helper__b1,r2,band_HIGH_PAIRED__condX2
        - 0.6181448971 aux_cap_helper__b1,r2,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b1,r2,band_HIGH_PAIRED__Z3  = 0
 c1430: X__b9,r4,band_UNPAIRED + X__b9,r4,band_HIGH_PAIRED
        - GlobalBidder_W_ir__b9,r4 + X__b9,r4,band_LOW_PAIRED >= 0
 c1431: aux_cap_b1,r7,band_UNPAIRED + aux_cap_b1,r7,band_LOW_PAIRED
        + aux_cap_b1,r7,band_HIGH_PAIRED - aux_c_b1,r7  = 0
 c1432: - aux_cap_helper__b4,r0,band_LOW_PAIRED__Z1
        + aux_cap_helper__b4,r0,band_LOW_PAIRED__condX1 <= 0
 c1433: - aux_cap_helper__b5,r11,band_UNPAIRED__condX2
        + aux_cap_helper__b5,r11,band_UNPAIRED__Z2 <= 0
 c1434: sv_function_helpervar_b0,r5__condX3
        - 15.9946187432626 sv_function_helpervar_b0,r5__Z3 <= 0
 c1435: - aux_cap_helper__b6,r4,band_LOW_PAIRED__Z1
        + aux_cap_helper__b6,r4,band_LOW_PAIRED__condX1 <= 0
 c1436: aux_cap_helper__b3,r9,band_LOW_PAIRED__Z1
        + aux_cap_helper__b3,r9,band_LOW_PAIRED__Z2  = 1
 c1437: - sv_function_helpervar_b7,r0__condX1 <= 0
 c1438: 6.04691642650674 sv_function_helpervar_b3,r0__Z3
        - sv_function_helpervar_b3,r0__condX3 <= 0
 c1439: aux_cap_helper__b1,r7,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b1,r7,band_LOW_PAIRED__Z2 <= 0
 c1440: aux_cap_helper__b9,r0,band_UNPAIRED__Z2
        + aux_cap_helper__b9,r0,band_UNPAIRED__Z1  = 1
 c1441: 2 aux_cap_helper__b4,r7,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b4,r7,band_HIGH_PAIRED__condX3 <= 0
 c1442: aux_cap_helper__b5,r12,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b5,r12,band_HIGH_PAIRED__Z1 <= 0
 c1443: X__b5,r4,band_UNPAIRED + X__b4,r4,band_UNPAIRED
        + X__b7,r4,band_UNPAIRED + X__b8,r4,band_UNPAIRED
        + X__b9,r4,band_UNPAIRED + X__b3,r4,band_UNPAIRED
        + X__b0,r4,band_UNPAIRED + X__b2,r4,band_UNPAIRED
        + X__b1,r4,band_UNPAIRED + X__b6,r4,band_UNPAIRED <= 2
 c1444: aux_cap_helper__b6,r5,band_UNPAIRED__Z1
        + aux_cap_helper__b6,r5,band_UNPAIRED__Z2  = 1
 c1445: 0.927217345653617 aux_cap_helper__b3,r9,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b3,r9,band_HIGH_PAIRED__Z2
        + 1.3908260184 aux_cap_helper__b3,r9,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b3,r9,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b3,r9,band_HIGH_PAIRED__condX1
        - aux_cap_b3,r9,band_HIGH_PAIRED  = 0
 c1446: aux_cap_helper__b5,r11,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b5,r11,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b5,r11,band_HIGH_PAIRED__Z1  = 1
 c1447: aux_cap_helper__b6,r2,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b6,r2,band_HIGH_PAIRED__Z2 <= 0
 c1448: - aux_cap_b9,r7,band_LOW_PAIRED
        - 1.4943850124 aux_cap_helper__b9,r7,band_LOW_PAIRED__Z2
        + 3.735962531 aux_cap_helper__b9,r7,band_LOW_PAIRED__condX1
        + 5.2303475434 aux_cap_helper__b9,r7,band_LOW_PAIRED__condX2  = 0
 c1449: aux_cap_b9,r12,band_HIGH_PAIRED + aux_cap_b9,r12,band_UNPAIRED
        - aux_c_b9,r12 + aux_cap_b9,r12,band_LOW_PAIRED  = 0
 c1450: - 6.10406421692187 sv_function_helpervar_b5,r11__Z2
        + sv_function_helpervar_b5,r11__condX2 <= 0
 c1451: aux_cap_helper__b2,r9,band_LOW_PAIRED__Z2
        - aux_cap_helper__b2,r9,band_LOW_PAIRED__condX2 <= 0
 c1452: - 15.9946187432626 sv_function_helpervar_b0,r11__Z3
        + sv_function_helpervar_b0,r11__condX3 <= 0
 c1453: - aux_cap_helper__b8,r6,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b8,r6,band_HIGH_PAIRED__condX1 <= 0
 c1454: - aux_c_b0,r12 + sv_function_helpervar_b0,r12__condX2
        + sv_function_helpervar_b0,r12__condX3
        + sv_function_helpervar_b0,r12__condX1  = 0
 c1455: sv_function_helpervar_b2,r8__Z1 + sv_function_helpervar_b2,r8__Z2
        + sv_function_helpervar_b2,r8__Z3  = 1
 c1456: aux_cap_helper__b2,r9,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b2,r9,band_HIGH_PAIRED__Z3 <= 0
 c1457: - aux_cap_helper__b8,r4,band_UNPAIRED__condX2
        + aux_cap_helper__b8,r4,band_UNPAIRED__Z2 <= 0
 c1458: aux_cap_helper__b0,r13,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b0,r13,band_HIGH_PAIRED__Z3 <= 0
 c1459: 3.735962531 aux_cap_helper__b5,r10,band_LOW_PAIRED__condX1
        + 5.2303475434 aux_cap_helper__b5,r10,band_LOW_PAIRED__condX2
        - 1.4943850124 aux_cap_helper__b5,r10,band_LOW_PAIRED__Z2
        - aux_cap_b5,r10,band_LOW_PAIRED  = 0
 c1460: - aux_sv_b5,r13 + 381.545981885315 sv_function_helpervar_b5,r13__Z3
        + 178.881328119173 sv_function_helpervar_b5,r13__Z2
        + 1169.5974421279 sv_function_helpervar_b5,r13__condX1
        + 19.4637769275 sv_function_helpervar_b5,r13__condX3
        + 51.2129970653 sv_function_helpervar_b5,r13__condX2  = 0
 c1461: - aux_cap_helper__b2,r9,band_LOW_PAIRED__condX1 <= 0
 c1462: aux_cap_helper__b3,r2,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b3,r2,band_HIGH_PAIRED__Z2 <= 0
 c1463: aux_cap_helper__b2,r13,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b2,r13,band_HIGH_PAIRED__condX2 <= 0
 c1464: - aux_cap_helper__b7,r12,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b7,r12,band_HIGH_PAIRED__Z2 <= 0
 c1465: - sv_function_helpervar_b7,r5__condX3
        + 6.42469180122383 sv_function_helpervar_b7,r5__Z3 <= 0
 c1466: aux_cap_helper__b3,r5,band_LOW_PAIRED__condX2
        + aux_cap_helper__b3,r5,band_LOW_PAIRED__condX1
        - X__b3,r5,band_LOW_PAIRED  = 0
 c1467: - aux_cap_helper__b2,r5,band_LOW_PAIRED__Z1
        + aux_cap_helper__b2,r5,band_LOW_PAIRED__condX1 <= 0
 c1468: - aux_cap_helper__b7,r7,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b7,r7,band_HIGH_PAIRED__Z3 <= 0
 c1469: aux_cap_helper__b5,r6,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b5,r6,band_LOW_PAIRED__Z2 <= 0
 c1470: aux_cap_helper__b3,r2,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b3,r2,band_UNPAIRED__Z2 <= 0
 c1471: - aux_cap_b9,r6,band_LOW_PAIRED
        - 1.4943850124 aux_cap_helper__b9,r6,band_LOW_PAIRED__Z2
        + 3.735962531 aux_cap_helper__b9,r6,band_LOW_PAIRED__condX1
        + 5.2303475434 aux_cap_helper__b9,r6,band_LOW_PAIRED__condX2  = 0
 c1472: - aux_cap_b6,r9,band_HIGH_PAIRED
        - 0.6181448971 aux_cap_helper__b6,r9,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b6,r9,band_HIGH_PAIRED__Z3
        + 1.3908260184 aux_cap_helper__b6,r9,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b6,r9,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b6,r9,band_HIGH_PAIRED__condX1  = 0
 c1473: - sv_function_helpervar_b8,r3__condX1 <= 0
 c1474: aux_cap_helper__b8,r13,band_UNPAIRED__Z2
        + aux_cap_helper__b8,r13,band_UNPAIRED__Z1  = 1
 c1475: - 0.159946187449246 sv_function_helpervar_b3,r6__Z1
        + sv_function_helpervar_b3,r6__condX1 <= 0
 c1476: sv_function_helpervar_b4,r5__Z1 + sv_function_helpervar_b4,r5__Z2
        + sv_function_helpervar_b4,r5__Z3  = 1
 c1477: - aux_cap_helper__b5,r11,band_UNPAIRED__condX1 <= 0
 c1478: - 15.9946187432626 sv_function_helpervar_b7,r9__Z3
        + sv_function_helpervar_b7,r9__condX3 <= 0
 c1479: - aux_cap_b4,r11,band_LOW_PAIRED
        - 1.4943850124 aux_cap_helper__b4,r11,band_LOW_PAIRED__Z2
        + 3.735962531 aux_cap_helper__b4,r11,band_LOW_PAIRED__condX1
        + 5.2303475434 aux_cap_helper__b4,r11,band_LOW_PAIRED__condX2  = 0
 c1480: sv_function_helpervar_b9,r7__condX2
        - 7.41632481705671 sv_function_helpervar_b9,r7__Z2 <= 0
 c1481: aux_cap_helper__b3,r7,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b3,r7,band_UNPAIRED__Z2 <= 0
 c1482: aux_cap_helper__b3,r5,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b3,r5,band_HIGH_PAIRED__Z2 <= 0
 c1483: aux_cap_helper__b2,r0,band_UNPAIRED__Z2
        + aux_cap_helper__b2,r0,band_UNPAIRED__Z1  = 1
 c1484: 0.927217345653617 aux_cap_helper__b2,r8,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b2,r8,band_HIGH_PAIRED__Z2
        - aux_cap_b2,r8,band_HIGH_PAIRED
        + 1.5453622427 aux_cap_helper__b2,r8,band_HIGH_PAIRED__condX1
        + 2.1635071398 aux_cap_helper__b2,r8,band_HIGH_PAIRED__condX2
        + 1.3908260184 aux_cap_helper__b2,r8,band_HIGH_PAIRED__condX3  = 0
 c1485: sv_function_helpervar_b7,r11__condX3
        - 15.9946187432626 sv_function_helpervar_b7,r11__Z3 <= 0
 c1486: - 5.65753539715363 sv_function_helpervar_b1,r5__Z2
        + sv_function_helpervar_b1,r5__condX2 <= 0
 c1487: aux_cap_b4,r6,band_LOW_PAIRED + aux_cap_b4,r6,band_HIGH_PAIRED
        - aux_c_b4,r6 + aux_cap_b4,r6,band_UNPAIRED  = 0
 c1488: - 2 aux_cap_helper__b6,r7,band_UNPAIRED__Z2
        + aux_cap_helper__b6,r7,band_UNPAIRED__condX2 <= 0
 c1489: aux_cap_b5,r11,band_UNPAIRED + aux_cap_b5,r11,band_LOW_PAIRED
        + aux_cap_b5,r11,band_HIGH_PAIRED - aux_c_b5,r11  = 0
 c1490: 2 aux_cap_helper__b7,r4,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b7,r4,band_HIGH_PAIRED__condX3 <= 0
 c1491: aux_cap_helper__b2,r6,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b2,r6,band_HIGH_PAIRED__Z3 <= 0
 c1492: aux_cap_helper__b0,r10,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b0,r10,band_HIGH_PAIRED__Z3 <= 0
 c1493: 1289.3272634054 sv_function_helpervar_b4,r2__condX1
        + 64.6044383885 sv_function_helpervar_b4,r2__condX2
        + 19.8361853949 sv_function_helpervar_b4,r2__condX3 - aux_sv_b4,r2
        + 446.516592701261 sv_function_helpervar_b4,r2__Z3
        + 195.889746548517 sv_function_helpervar_b4,r2__Z2  = 0
 c1494: aux_cap_b9,r8,band_UNPAIRED + aux_cap_b9,r8,band_LOW_PAIRED
        + aux_cap_b9,r8,band_HIGH_PAIRED - aux_c_b9,r8  = 0
 c1495: - sv_function_helpervar_b0,r13__condX1 <= 0
 c1496: aux_cap_helper__b9,r8,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b9,r8,band_HIGH_PAIRED__condX2 <= 0
 c1497: aux_cap_helper__b0,r2,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b0,r2,band_UNPAIRED__Z2 <= 0
 c1498: - 0.159946187506737 sv_function_helpervar_b0,r7__Z1
        + sv_function_helpervar_b0,r7__condX1 <= 0
 c1499: 1.1250244065 aux_cap_helper__b2,r0,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b2,r0,band_UNPAIRED__condX1
        - aux_cap_b2,r0,band_UNPAIRED
        - 0.3214355447 aux_cap_helper__b2,r0,band_UNPAIRED__Z2  = 0
 c1500: 5.2303475434 aux_cap_helper__b0,r8,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b0,r8,band_LOW_PAIRED__condX1
        - aux_cap_b0,r8,band_LOW_PAIRED
        - 1.4943850124 aux_cap_helper__b0,r8,band_LOW_PAIRED__Z2  = 0
 c1501: aux_cap_helper__b0,r8,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b0,r8,band_HIGH_PAIRED__Z2 <= 0
 c1502: - aux_cap_helper__b8,r9,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b8,r9,band_HIGH_PAIRED__condX1 <= 0
 c1503: 0.0102040816326531 X__b8,r7,band_LOW_PAIRED
        + 0.0102040816326531 X__b8,r7,band_UNPAIRED - GlobalBidder_W_ir__b8,r7
        + 0.0102040816326531 X__b8,r7,band_HIGH_PAIRED <= 0
 c1504: - aux_cap_b9,r9,band_HIGH_PAIRED
        + 2.1635071398 aux_cap_helper__b9,r9,band_HIGH_PAIRED__condX2
        + 1.3908260184 aux_cap_helper__b9,r9,band_HIGH_PAIRED__condX3
        + 1.5453622427 aux_cap_helper__b9,r9,band_HIGH_PAIRED__condX1
        + 0.927217345653617 aux_cap_helper__b9,r9,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b9,r9,band_HIGH_PAIRED__Z2  = 0
 c1505: - aux_cap_helper__b9,r8,band_HIGH_PAIRED__condX1 <= 0
 c1506: sv_function_helpervar_b9,r10__Z3 + sv_function_helpervar_b9,r10__Z1
        + sv_function_helpervar_b9,r10__Z2  = 1
 c1507: - aux_cap_helper__b0,r3,band_LOW_PAIRED__condX2
        + aux_cap_helper__b0,r3,band_LOW_PAIRED__Z2 <= 0
 c1508: sv_function_helpervar_b9,r11__condX3
        - 15.9946187432626 sv_function_helpervar_b9,r11__Z3 <= 0
 c1509: - X__b5,r11,band_LOW_PAIRED
        + aux_cap_helper__b5,r11,band_LOW_PAIRED__condX2
        + aux_cap_helper__b5,r11,band_LOW_PAIRED__condX1  = 0
 c1510: - X__b7,r9,band_HIGH_PAIRED
        + aux_cap_helper__b7,r9,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b7,r9,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b7,r9,band_HIGH_PAIRED__condX1  = 0
 c1511: - 0.3214355447 aux_cap_helper__b6,r5,band_UNPAIRED__Z2
        + 0.8035888618 aux_cap_helper__b6,r5,band_UNPAIRED__condX1
        + 1.1250244065 aux_cap_helper__b6,r5,band_UNPAIRED__condX2
        - aux_cap_b6,r5,band_UNPAIRED  = 0
 c1512: 1.1250244065 aux_cap_helper__b3,r3,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b3,r3,band_UNPAIRED__condX1
        - aux_cap_b3,r3,band_UNPAIRED
        - 0.3214355447 aux_cap_helper__b3,r3,band_UNPAIRED__Z2  = 0
 c1513: 0.0102040816326531 X__b9,r12,band_LOW_PAIRED
        - GlobalBidder_W_ir__b9,r12
        + 0.0102040816326531 X__b9,r12,band_UNPAIRED
        + 0.0102040816326531 X__b9,r12,band_HIGH_PAIRED <= 0
 c1514: - 0.3214355447 aux_cap_helper__b0,r1,band_UNPAIRED__Z2
        - aux_cap_b0,r1,band_UNPAIRED
        + 0.8035888618 aux_cap_helper__b0,r1,band_UNPAIRED__condX1
        + 1.1250244065 aux_cap_helper__b0,r1,band_UNPAIRED__condX2  = 0
 c1515: aux_cap_helper__b1,r2,band_LOW_PAIRED__condX1
        - aux_cap_helper__b1,r2,band_LOW_PAIRED__Z1 <= 0
 c1516: - aux_cap_helper__b1,r12,band_HIGH_PAIRED__condX1 <= 0
 c1517: aux_cap_helper__b2,r5,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b2,r5,band_HIGH_PAIRED__Z3 <= 0
 c1518: aux_cap_helper__b6,r4,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b6,r4,band_HIGH_PAIRED__Z1 <= 0
 c1519: - sv_function_helpervar_b9,r10__condX2
        + 0.159946187433718 sv_function_helpervar_b9,r10__Z2 <= 0
 c1520: - X__b8,r11,band_HIGH_PAIRED
        + aux_cap_helper__b8,r11,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b8,r11,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b8,r11,band_HIGH_PAIRED__condX2  = 0
 c1521: 2 aux_cap_helper__b7,r1,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b7,r1,band_HIGH_PAIRED__condX3 <= 0
 c1522: aux_cap_helper__b9,r0,band_UNPAIRED__condX1
        - aux_cap_helper__b9,r0,band_UNPAIRED__Z1 <= 0
 c1523: - aux_cap_helper__b2,r4,band_UNPAIRED__condX1 <= 0
 c1524: - sv_function_helpervar_b2,r8__condX2
        + 0.159946187457799 sv_function_helpervar_b2,r8__Z2 <= 0
 c1525: aux_cap_helper__b6,r4,band_LOW_PAIRED__Z1
        + aux_cap_helper__b6,r4,band_LOW_PAIRED__Z2  = 1
 c1526: sv_function_helpervar_b1,r12__Z3 + sv_function_helpervar_b1,r12__Z2
        + sv_function_helpervar_b1,r12__Z1  = 1
 c1527: aux_cap_helper__b4,r13,band_UNPAIRED__Z2
        - aux_cap_helper__b4,r13,band_UNPAIRED__condX2 <= 0
 c1528: aux_Omega_b9,r1 + aux_Omega_b9,r0 + aux_Omega_b9,r3 + aux_Omega_b9,r2
        + aux_Omega_b9,r5 + aux_Omega_b9,r4 + aux_Omega_b9,r7 + aux_Omega_b9,r6
        + aux_Omega_b9,r9 + aux_Omega_b9,r8 - GlobalBidder_PSI__i9,k0
        + aux_Omega_b9,r13 + aux_Omega_b9,r10 + aux_Omega_b9,r11
        + aux_Omega_b9,r12 - 18497526.8940961 GlobalBidder_hat{W}__i9,k0
        >= -18497526.8940961
 c1529: 0.0102040816326531 X__b9,r9,band_LOW_PAIRED
        + 0.0102040816326531 X__b9,r9,band_UNPAIRED
        + 0.0102040816326531 X__b9,r9,band_HIGH_PAIRED
        - GlobalBidder_W_ir__b9,r9 <= 0
 c1530: 4090.2281771502 aux_sv_b9,r11 - aux_Omega_b9,r11  = 0
 c1531: aux_cap_helper__b5,r2,band_LOW_PAIRED__condX2
        + aux_cap_helper__b5,r2,band_LOW_PAIRED__condX1
        - X__b5,r2,band_LOW_PAIRED  = 0
 c1532: - 2 aux_cap_helper__b8,r13,band_LOW_PAIRED__Z2
        + aux_cap_helper__b8,r13,band_LOW_PAIRED__condX2 <= 0
 c1533: - aux_cap_b7,r7,band_LOW_PAIRED
        - 1.4943850124 aux_cap_helper__b7,r7,band_LOW_PAIRED__Z2
        + 5.2303475434 aux_cap_helper__b7,r7,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b7,r7,band_LOW_PAIRED__condX1  = 0
 c1534: aux_cap_helper__b0,r5,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b0,r5,band_HIGH_PAIRED__Z2 <= 0
 c1535: aux_cap_helper__b8,r11,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b8,r11,band_UNPAIRED__Z2 <= 0
 c1536: aux_cap_helper__b5,r3,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b5,r3,band_LOW_PAIRED__Z2 <= 0
 c1537: - aux_cap_helper__b6,r10,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b6,r10,band_HIGH_PAIRED__Z3 <= 0
 c1538: - sv_function_helpervar_b7,r12__condX2
        + 0.159946187505732 sv_function_helpervar_b7,r12__Z2 <= 0
 c1539: sv_function_helpervar_b0,r5__condX2
        - 6.21712284501458 sv_function_helpervar_b0,r5__Z2 <= 0
 c1540: - aux_cap_b5,r8,band_UNPAIRED
        + 1.1250244065 aux_cap_helper__b5,r8,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b5,r8,band_UNPAIRED__condX1
        - 0.3214355447 aux_cap_helper__b5,r8,band_UNPAIRED__Z2  = 0
 c1541: 1.1250244065 aux_cap_helper__b2,r6,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b2,r6,band_UNPAIRED__condX1
        - aux_cap_b2,r6,band_UNPAIRED
        - 0.3214355447 aux_cap_helper__b2,r6,band_UNPAIRED__Z2  = 0
 c1542: - 1.4943850124 aux_cap_helper__b8,r9,band_LOW_PAIRED__Z2
        + 3.735962531 aux_cap_helper__b8,r9,band_LOW_PAIRED__condX1
        + 5.2303475434 aux_cap_helper__b8,r9,band_LOW_PAIRED__condX2
        - aux_cap_b8,r9,band_LOW_PAIRED  = 0
 c1543: - X__b5,r5,band_HIGH_PAIRED
        + aux_cap_helper__b5,r5,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b5,r5,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b5,r5,band_HIGH_PAIRED__condX2  = 0
 c1544: sv_function_helpervar_b1,r8__condX2
        + sv_function_helpervar_b1,r8__condX3
        + sv_function_helpervar_b1,r8__condX1 - aux_c_b1,r8  = 0
 c1545: - X__b5,r9,band_LOW_PAIRED
        + aux_cap_helper__b5,r9,band_LOW_PAIRED__condX2
        + aux_cap_helper__b5,r9,band_LOW_PAIRED__condX1  = 0
 c1546: aux_cap_helper__b9,r7,band_LOW_PAIRED__Z1
        + aux_cap_helper__b9,r7,band_LOW_PAIRED__Z2  = 1
 c1547: - sv_function_helpervar_b2,r9__condX1 <= 0
 c1548: aux_cap_helper__b1,r10,band_UNPAIRED__Z2
        - aux_cap_helper__b1,r10,band_UNPAIRED__condX2 <= 0
 c1549: - aux_Omega_b6,r4 + 722.628483855863 aux_sv_b6,r4  = 0
 c1550: - aux_cap_helper__b3,r0,band_UNPAIRED__condX1 <= 0
 c1551: aux_cap_helper__b7,r7,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b7,r7,band_UNPAIRED__Z2 <= 0
 c1552: - aux_cap_helper__b8,r4,band_UNPAIRED__condX1 <= 0
 c1553: aux_cap_b9,r2,band_LOW_PAIRED + aux_cap_b9,r2,band_UNPAIRED
        + aux_cap_b9,r2,band_HIGH_PAIRED - aux_c_b9,r2  = 0
 c1554: 69.493965406291 aux_sv_b9,r13 - aux_Omega_b9,r13  = 0
 c1555: - aux_cap_helper__b9,r9,band_UNPAIRED__condX1 <= 0
 c1556: - aux_cap_helper__b2,r4,band_UNPAIRED__condX2
        + aux_cap_helper__b2,r4,band_UNPAIRED__Z2 <= 0
 c1557: aux_cap_helper__b9,r10,band_UNPAIRED__Z1
        + aux_cap_helper__b9,r10,band_UNPAIRED__Z2  = 1
 c1558: - aux_cap_helper__b2,r10,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b2,r10,band_HIGH_PAIRED__Z2 <= 0
 c1559: aux_cap_helper__b8,r12,band_UNPAIRED__Z1
        + aux_cap_helper__b8,r12,band_UNPAIRED__Z2  = 1
 c1560: - aux_cap_helper__b0,r3,band_LOW_PAIRED__condX1 <= 0
 c1561: - aux_cap_helper__b4,r6,band_LOW_PAIRED__Z1
        + aux_cap_helper__b4,r6,band_LOW_PAIRED__condX1 <= 0
 c1562: - 2 aux_cap_helper__b5,r7,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b5,r7,band_HIGH_PAIRED__condX2 <= 0
 c1563: aux_cap_helper__b9,r6,band_LOW_PAIRED__Z2
        - aux_cap_helper__b9,r6,band_LOW_PAIRED__condX2 <= 0
 c1564: - aux_cap_helper__b2,r10,band_HIGH_PAIRED__condX1 <= 0
 c1565: 1.3908260184 aux_cap_helper__b2,r9,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b2,r9,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b2,r9,band_HIGH_PAIRED__condX1
        - 0.6181448971 aux_cap_helper__b2,r9,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b2,r9,band_HIGH_PAIRED__Z3
        - aux_cap_b2,r9,band_HIGH_PAIRED  = 0
 c1566: - aux_cap_b8,r4,band_LOW_PAIRED
        - 1.4943850124 aux_cap_helper__b8,r4,band_LOW_PAIRED__Z2
        + 3.735962531 aux_cap_helper__b8,r4,band_LOW_PAIRED__condX1
        + 5.2303475434 aux_cap_helper__b8,r4,band_LOW_PAIRED__condX2  = 0
 c1567: aux_cap_helper__b7,r9,band_UNPAIRED__Z1
        + aux_cap_helper__b7,r9,band_UNPAIRED__Z2  = 1
 c1568: 2 aux_cap_helper__b5,r2,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b5,r2,band_HIGH_PAIRED__condX3 <= 0
 c1569: - aux_cap_b6,r10,band_HIGH_PAIRED
        + 1.3908260184 aux_cap_helper__b6,r10,band_HIGH_PAIRED__condX3
        + 1.5453622427 aux_cap_helper__b6,r10,band_HIGH_PAIRED__condX1
        + 2.1635071398 aux_cap_helper__b6,r10,band_HIGH_PAIRED__condX2
        + 0.927217345653617 aux_cap_helper__b6,r10,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b6,r10,band_HIGH_PAIRED__Z2  = 0
 c1570: aux_cap_helper__b7,r8,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b7,r8,band_HIGH_PAIRED__Z3 <= 0
 c1571: sv_function_helpervar_b2,r5__condX2
        + sv_function_helpervar_b2,r5__condX3
        + sv_function_helpervar_b2,r5__condX1 - aux_c_b2,r5  = 0
 c1572: aux_cap_helper__b3,r12,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b3,r12,band_LOW_PAIRED__Z2 <= 0
 c1573: - 2 aux_cap_helper__b4,r13,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b4,r13,band_HIGH_PAIRED__condX2 <= 0
 c1574: - 15.9946187432626 sv_function_helpervar_b7,r8__Z3
        + sv_function_helpervar_b7,r8__condX3 <= 0
 c1575: 0.927217345653617 aux_cap_helper__b9,r0,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b9,r0,band_HIGH_PAIRED__Z2
        - aux_cap_b9,r0,band_HIGH_PAIRED
        + 1.3908260184 aux_cap_helper__b9,r0,band_HIGH_PAIRED__condX3
        + 1.5453622427 aux_cap_helper__b9,r0,band_HIGH_PAIRED__condX1
        + 2.1635071398 aux_cap_helper__b9,r0,band_HIGH_PAIRED__condX2  = 0
 c1576: - X__b4,r5,band_UNPAIRED + aux_cap_helper__b4,r5,band_UNPAIRED__condX2
        + aux_cap_helper__b4,r5,band_UNPAIRED__condX1  = 0
 c1577: - aux_cap_helper__b9,r6,band_LOW_PAIRED__condX1 <= 0
 c1578: aux_cap_helper__b8,r1,band_UNPAIRED__Z1
        + aux_cap_helper__b8,r1,band_UNPAIRED__Z2  = 1
 c1579: aux_cap_helper__b0,r6,band_LOW_PAIRED__Z1
        + aux_cap_helper__b0,r6,band_LOW_PAIRED__Z2  = 1
 c1580: X__b0,r4,band_LOW_PAIRED + X__b7,r4,band_LOW_PAIRED
        + X__b4,r4,band_LOW_PAIRED + X__b5,r4,band_LOW_PAIRED
        + X__b3,r4,band_LOW_PAIRED + X__b1,r4,band_LOW_PAIRED
        + X__b2,r4,band_LOW_PAIRED + X__b8,r4,band_LOW_PAIRED
        + X__b6,r4,band_LOW_PAIRED + X__b9,r4,band_LOW_PAIRED <= 2
 c1581: - aux_cap_helper__b1,r1,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b1,r1,band_HIGH_PAIRED__condX1 <= 0
 c1582: - aux_cap_helper__b3,r0,band_UNPAIRED__condX2
        + aux_cap_helper__b3,r0,band_UNPAIRED__Z2 <= 0
 c1583: aux_cap_b0,r3,band_UNPAIRED + aux_cap_b0,r3,band_LOW_PAIRED
        + aux_cap_b0,r3,band_HIGH_PAIRED - aux_c_b0,r3  = 0
 c1584: aux_cap_helper__b1,r12,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b1,r12,band_HIGH_PAIRED__condX2 <= 0
 c1585: - 6.71456150061446 sv_function_helpervar_b7,r10__Z2
        + sv_function_helpervar_b7,r10__condX2 <= 0
 c1586: sv_function_helpervar_b3,r9__Z1 + sv_function_helpervar_b3,r9__Z2
        + sv_function_helpervar_b3,r9__Z3  = 1
 c1587: - 2 aux_cap_helper__b2,r13,band_LOW_PAIRED__Z2
        + aux_cap_helper__b2,r13,band_LOW_PAIRED__condX2 <= 0
 c1588: - sv_function_helpervar_b9,r2__condX1 <= 0
 c1589: 0.159946187436826 sv_function_helpervar_b4,r3__Z2
        - sv_function_helpervar_b4,r3__condX2 <= 0
 c1590: - aux_Omega_b6,r3 + 399.456464833497 aux_sv_b6,r3  = 0
 c1591: aux_cap_helper__b5,r0,band_LOW_PAIRED__Z2
        + aux_cap_helper__b5,r0,band_LOW_PAIRED__Z1  = 1
 c1592: 559.630509527518 sv_function_helpervar_b9,r9__Z3
        + 262.970208465602 sv_function_helpervar_b9,r9__Z2
        + 1719.0945882549 sv_function_helpervar_b9,r9__condX1
        + 74.9778219796 sv_function_helpervar_b9,r9__condX2
        + 28.6814954013 sv_function_helpervar_b9,r9__condX3 - aux_sv_b9,r9  = 0
 c1593: aux_cap_helper__b1,r8,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b1,r8,band_UNPAIRED__Z2 <= 0
 c1594: aux_cap_helper__b1,r7,band_UNPAIRED__condX1
        - aux_cap_helper__b1,r7,band_UNPAIRED__Z1 <= 0
 c1595: - aux_cap_helper__b8,r13,band_HIGH_PAIRED__condX1 <= 0
 c1596: 2.1635071398 aux_cap_helper__b0,r8,band_HIGH_PAIRED__condX2
        + 1.3908260184 aux_cap_helper__b0,r8,band_HIGH_PAIRED__condX3
        + 1.5453622427 aux_cap_helper__b0,r8,band_HIGH_PAIRED__condX1
        - aux_cap_b0,r8,band_HIGH_PAIRED
        - 0.6181448971 aux_cap_helper__b0,r8,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b0,r8,band_HIGH_PAIRED__Z3  = 0
 c1597: - X__b5,r8,band_UNPAIRED + aux_cap_helper__b5,r8,band_UNPAIRED__condX2
        + aux_cap_helper__b5,r8,band_UNPAIRED__condX1  = 0
 c1598: - aux_cap_helper__b4,r5,band_HIGH_PAIRED__condX1 <= 0
 c1599: - sv_function_helpervar_b8,r4__condX1 <= 0
 c1600: 3.735962531 aux_cap_helper__b9,r10,band_LOW_PAIRED__condX1
        + 5.2303475434 aux_cap_helper__b9,r10,band_LOW_PAIRED__condX2
        - 1.4943850124 aux_cap_helper__b9,r10,band_LOW_PAIRED__Z2
        - aux_cap_b9,r10,band_LOW_PAIRED  = 0
 c1601: sv_function_helpervar_b3,r1__Z3 + sv_function_helpervar_b3,r1__Z2
        + sv_function_helpervar_b3,r1__Z1  = 1
 c1602: sv_function_helpervar_b0,r12__Z2 + sv_function_helpervar_b0,r12__Z3
        + sv_function_helpervar_b0,r12__Z1  = 1
 c1603: - 5.73573107720359 sv_function_helpervar_b6,r0__Z2
        + sv_function_helpervar_b6,r0__condX2 <= 0
 c1604: - sv_function_helpervar_b6,r13__condX1 <= 0
 c1605: aux_cap_helper__b0,r6,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b0,r6,band_HIGH_PAIRED__Z3 <= 0
 c1606: aux_cap_helper__b5,r1,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b5,r1,band_UNPAIRED__Z2 <= 0
 c1607: - 3 aux_cap_helper__b8,r5,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b8,r5,band_HIGH_PAIRED__condX3 <= 0
 c1608: - aux_cap_b0,r0,band_LOW_PAIRED
        - 1.4943850124 aux_cap_helper__b0,r0,band_LOW_PAIRED__Z2
        + 5.2303475434 aux_cap_helper__b0,r0,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b0,r0,band_LOW_PAIRED__condX1  = 0
 c1609: - 0.6181448971 aux_cap_helper__b1,r8,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b1,r8,band_HIGH_PAIRED__Z3
        + 1.5453622427 aux_cap_helper__b1,r8,band_HIGH_PAIRED__condX1
        + 2.1635071398 aux_cap_helper__b1,r8,band_HIGH_PAIRED__condX2
        + 1.3908260184 aux_cap_helper__b1,r8,band_HIGH_PAIRED__condX3
        - aux_cap_b1,r8,band_HIGH_PAIRED  = 0
 c1610: - aux_Omega_b6,r11 + 2074.30794707567 aux_sv_b6,r11  = 0
 c1611: - 15.9946187432626 sv_function_helpervar_b1,r3__Z3
        + sv_function_helpervar_b1,r3__condX3 <= 0
 c1612: - 7.02419321281872 sv_function_helpervar_b9,r12__Z2
        + sv_function_helpervar_b9,r12__condX2 <= 0
 c1613: aux_cap_helper__b5,r2,band_UNPAIRED__condX1
        - aux_cap_helper__b5,r2,band_UNPAIRED__Z1 <= 0
 c1614: aux_cap_helper__b6,r7,band_LOW_PAIRED__condX1
        - aux_cap_helper__b6,r7,band_LOW_PAIRED__Z1 <= 0
 c1615: aux_cap_helper__b9,r9,band_UNPAIRED__Z2
        - aux_cap_helper__b9,r9,band_UNPAIRED__condX2 <= 0
 c1616: aux_cap_b1,r1,band_UNPAIRED + aux_cap_b1,r1,band_HIGH_PAIRED
        + aux_cap_b1,r1,band_LOW_PAIRED - aux_c_b1,r1  = 0
 c1617: aux_cap_helper__b0,r5,band_LOW_PAIRED__Z2
        - aux_cap_helper__b0,r5,band_LOW_PAIRED__condX2 <= 0
 c1618: - 0.159946187486893 sv_function_helpervar_b5,r2__Z1
        + sv_function_helpervar_b5,r2__condX1 <= 0
 c1619: aux_cap_helper__b1,r11,band_LOW_PAIRED__condX2
        + aux_cap_helper__b1,r11,band_LOW_PAIRED__condX1
        - X__b1,r11,band_LOW_PAIRED  = 0
 c1620: aux_cap_helper__b4,r6,band_UNPAIRED__condX1
        - aux_cap_helper__b4,r6,band_UNPAIRED__Z1 <= 0
 c1621: aux_cap_helper__b1,r2,band_LOW_PAIRED__Z2
        + aux_cap_helper__b1,r2,band_LOW_PAIRED__Z1  = 1
 c1622: - aux_cap_b8,r0,band_HIGH_PAIRED
        + 1.3908260184 aux_cap_helper__b8,r0,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b8,r0,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b8,r0,band_HIGH_PAIRED__condX1
        + 0.927217345653617 aux_cap_helper__b8,r0,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b8,r0,band_HIGH_PAIRED__Z2  = 0
 c1623: sv_function_helpervar_b3,r3__condX2
        - 5.6831725135299 sv_function_helpervar_b3,r3__Z2 <= 0
 c1624: - 2 aux_cap_helper__b5,r8,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b5,r8,band_HIGH_PAIRED__condX2 <= 0
 c1625: - sv_function_helpervar_b8,r0__condX2
        + 0.159946187434592 sv_function_helpervar_b8,r0__Z2 <= 0
 c1626: - X__b8,r2,band_LOW_PAIRED
        + aux_cap_helper__b8,r2,band_LOW_PAIRED__condX2
        + aux_cap_helper__b8,r2,band_LOW_PAIRED__condX1  = 0
 c1627: - aux_cap_helper__b6,r0,band_LOW_PAIRED__condX2
        + aux_cap_helper__b6,r0,band_LOW_PAIRED__Z2 <= 0
 c1628: 1.5453622427 aux_cap_helper__b6,r2,band_HIGH_PAIRED__condX1
        + 1.3908260184 aux_cap_helper__b6,r2,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b6,r2,band_HIGH_PAIRED__condX2
        - aux_cap_b6,r2,band_HIGH_PAIRED
        - 0.6181448971 aux_cap_helper__b6,r2,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b6,r2,band_HIGH_PAIRED__Z3  = 0
 c1629: - aux_cap_helper__b0,r5,band_LOW_PAIRED__condX1 <= 0
 c1630: - aux_cap_helper__b9,r1,band_LOW_PAIRED__condX1 <= 0
 c1631: - aux_cap_b1,r9,band_HIGH_PAIRED
        + 1.3908260184 aux_cap_helper__b1,r9,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b1,r9,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b1,r9,band_HIGH_PAIRED__condX1
        - 0.6181448971 aux_cap_helper__b1,r9,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b1,r9,band_HIGH_PAIRED__Z3  = 0
 c1632: aux_cap_helper__b4,r5,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b4,r5,band_HIGH_PAIRED__condX2 <= 0
 c1633: - sv_function_helpervar_b5,r11__condX1 <= 0
 c1634: aux_cap_helper__b2,r5,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b2,r5,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b2,r5,band_HIGH_PAIRED__Z2  = 1
 c1635: 497.270807787582 aux_sv_b8,r2 - aux_Omega_b8,r2  = 0
 c1636: - X__b2,r3,band_LOW_PAIRED
        + aux_cap_helper__b2,r3,band_LOW_PAIRED__condX1
        + aux_cap_helper__b2,r3,band_LOW_PAIRED__condX2  = 0
 c1637: 0.159946187462122 sv_function_helpervar_b4,r6__Z2
        - sv_function_helpervar_b4,r6__condX2 <= 0
 c1638: - 0.1599461874636 sv_function_helpervar_b1,r1__Z1
        + sv_function_helpervar_b1,r1__condX1 <= 0
 c1639: 2 aux_cap_helper__b5,r1,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b5,r1,band_HIGH_PAIRED__condX3 <= 0
 c1640: - 15.9946187432626 sv_function_helpervar_b2,r13__Z3
        + sv_function_helpervar_b2,r13__condX3 <= 0
 c1641: - sv_function_helpervar_b5,r5__condX2
        + 0.159946187436384 sv_function_helpervar_b5,r5__Z2 <= 0
 c1642: - 3 aux_cap_helper__b0,r0,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b0,r0,band_HIGH_PAIRED__condX3 <= 0
 c1643: 5.2303475434 aux_cap_helper__b2,r7,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b2,r7,band_LOW_PAIRED__condX1
        - 1.4943850124 aux_cap_helper__b2,r7,band_LOW_PAIRED__Z2
        - aux_cap_b2,r7,band_LOW_PAIRED  = 0
 c1644: - aux_cap_b0,r11,band_UNPAIRED
        - 0.3214355447 aux_cap_helper__b0,r11,band_UNPAIRED__Z2
        + 1.1250244065 aux_cap_helper__b0,r11,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b0,r11,band_UNPAIRED__condX1  = 0
 c1645: - 6.13038020413258 sv_function_helpervar_b1,r10__Z2
        + sv_function_helpervar_b1,r10__condX2 <= 0
 c1646: - sv_function_helpervar_b9,r10__condX3
        + 7.70903696793851 sv_function_helpervar_b9,r10__Z3 <= 0
 c1647: sv_function_helpervar_b3,r10__condX3
        + sv_function_helpervar_b3,r10__condX1
        + sv_function_helpervar_b3,r10__condX2 - aux_c_b3,r10  = 0
 c1648: aux_cap_helper__b3,r4,band_UNPAIRED__Z1
        + aux_cap_helper__b3,r4,band_UNPAIRED__Z2  = 1
 c1649: 35.7228892553 sv_function_helpervar_b7,r12__condX3
        + 66.783347368 sv_function_helpervar_b7,r12__condX2
        + 1850.3233251435 sv_function_helpervar_b7,r12__condX1
        + 524.745122656364 sv_function_helpervar_b7,r12__Z3
        + 285.27041970925 sv_function_helpervar_b7,r12__Z2 - aux_sv_b7,r12  = 0
 c1650: - aux_cap_helper__b6,r3,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b6,r3,band_HIGH_PAIRED__condX1 <= 0
 c1651: - aux_cap_helper__b2,r3,band_UNPAIRED__Z1
        + aux_cap_helper__b2,r3,band_UNPAIRED__condX1 <= 0
 c1652: - aux_Omega_b7,r0 + 284.531149003662 aux_sv_b7,r0  = 0
 c1653: aux_cap_helper__b8,r13,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b8,r13,band_HIGH_PAIRED__condX2 <= 0
 c1654: - 0.3214355447 aux_cap_helper__b2,r8,band_UNPAIRED__Z2
        - aux_cap_b2,r8,band_UNPAIRED
        + 1.1250244065 aux_cap_helper__b2,r8,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b2,r8,band_UNPAIRED__condX1  = 0
 c1655: - 0.6181448971 aux_cap_helper__b7,r3,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b7,r3,band_HIGH_PAIRED__Z3
        + 1.3908260184 aux_cap_helper__b7,r3,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b7,r3,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b7,r3,band_HIGH_PAIRED__condX1
        - aux_cap_b7,r3,band_HIGH_PAIRED  = 0
 c1656: - aux_c_b6,r11 + aux_cap_b6,r11,band_HIGH_PAIRED
        + aux_cap_b6,r11,band_LOW_PAIRED + aux_cap_b6,r11,band_UNPAIRED  = 0
 c1657: aux_cap_helper__b0,r3,band_LOW_PAIRED__condX1
        + aux_cap_helper__b0,r3,band_LOW_PAIRED__condX2
        - X__b0,r3,band_LOW_PAIRED  = 0
 c1658: 0.927217345653617 aux_cap_helper__b6,r3,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b6,r3,band_HIGH_PAIRED__Z2
        + 1.3908260184 aux_cap_helper__b6,r3,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b6,r3,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b6,r3,band_HIGH_PAIRED__condX1
        - aux_cap_b6,r3,band_HIGH_PAIRED  = 0
 c1659: sv_function_helpervar_b8,r10__condX1
        - 0.159946187439248 sv_function_helpervar_b8,r10__Z1 <= 0
 c1660: aux_cap_helper__b3,r9,band_LOW_PAIRED__Z2
        - aux_cap_helper__b3,r9,band_LOW_PAIRED__condX2 <= 0
 c1661: - aux_cap_helper__b6,r11,band_UNPAIRED__condX2
        + aux_cap_helper__b6,r11,band_UNPAIRED__Z2 <= 0
 c1662: - 2 aux_cap_helper__b8,r10,band_LOW_PAIRED__Z2
        + aux_cap_helper__b8,r10,band_LOW_PAIRED__condX2 <= 0
 c1663: X__b3,r10,band_UNPAIRED + X__b9,r10,band_UNPAIRED
        + X__b1,r10,band_UNPAIRED + X__b5,r10,band_UNPAIRED
        + X__b7,r10,band_UNPAIRED + X__b2,r10,band_UNPAIRED
        + X__b0,r10,band_UNPAIRED + X__b6,r10,band_UNPAIRED
        + X__b8,r10,band_UNPAIRED + X__b4,r10,band_UNPAIRED <= 2
 c1664: aux_cap_helper__b3,r10,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b3,r10,band_HIGH_PAIRED__Z1 <= 0
 c1665: - 2 aux_cap_helper__b4,r10,band_UNPAIRED__Z2
        + aux_cap_helper__b4,r10,band_UNPAIRED__condX2 <= 0
 c1666: sv_function_helpervar_b8,r0__condX2
        - 6.25776005044761 sv_function_helpervar_b8,r0__Z2 <= 0
 c1667: - aux_cap_helper__b8,r7,band_UNPAIRED__Z1
        + aux_cap_helper__b8,r7,band_UNPAIRED__condX1 <= 0
 c1668: - aux_cap_helper__b0,r8,band_UNPAIRED__Z1
        + aux_cap_helper__b0,r8,band_UNPAIRED__condX1 <= 0
 c1669: 0.0102040816326531 X__b7,r3,band_LOW_PAIRED - GlobalBidder_W_ir__b7,r3
        + 0.0102040816326531 X__b7,r3,band_UNPAIRED
        + 0.0102040816326531 X__b7,r3,band_HIGH_PAIRED <= 0
 c1670: 0.927217345653617 aux_cap_helper__b8,r9,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b8,r9,band_HIGH_PAIRED__Z2
        - aux_cap_b8,r9,band_HIGH_PAIRED
        + 1.3908260184 aux_cap_helper__b8,r9,band_HIGH_PAIRED__condX3
        + 1.5453622427 aux_cap_helper__b8,r9,band_HIGH_PAIRED__condX1
        + 2.1635071398 aux_cap_helper__b8,r9,band_HIGH_PAIRED__condX2  = 0
 c1671: aux_cap_helper__b1,r11,band_UNPAIRED__condX1
        + aux_cap_helper__b1,r11,band_UNPAIRED__condX2
        - X__b1,r11,band_UNPAIRED  = 0
 c1672: - 3 aux_cap_helper__b3,r0,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b3,r0,band_HIGH_PAIRED__condX3 <= 0
 c1673: aux_cap_helper__b4,r2,band_UNPAIRED__condX1
        - aux_cap_helper__b4,r2,band_UNPAIRED__Z1 <= 0
 c1674: - sv_function_helpervar_b8,r4__condX3
        + 8.24097916477441 sv_function_helpervar_b8,r4__Z3 <= 0
 c1675: aux_cap_helper__b2,r6,band_LOW_PAIRED__Z2
        + aux_cap_helper__b2,r6,band_LOW_PAIRED__Z1  = 1
 c1676: - 0.6181448971 aux_cap_helper__b7,r9,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b7,r9,band_HIGH_PAIRED__Z3
        - aux_cap_b7,r9,band_HIGH_PAIRED
        + 1.3908260184 aux_cap_helper__b7,r9,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b7,r9,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b7,r9,band_HIGH_PAIRED__condX1  = 0
 c1677: 32.0532995954 sv_function_helpervar_b9,r7__condX3
        + 64.5576625617 sv_function_helpervar_b9,r7__condX2
        + 1719.0945882664 sv_function_helpervar_b9,r7__condX1
        + 505.699786966617 sv_function_helpervar_b9,r7__Z3
        + 264.636873237029 sv_function_helpervar_b9,r7__Z2 - aux_sv_b9,r7  = 0
 c1678: aux_cap_helper__b5,r4,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b5,r4,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b5,r4,band_HIGH_PAIRED__Z1  = 1
 c1679: 6.50772434820002 sv_function_helpervar_b0,r0__Z3
        - sv_function_helpervar_b0,r0__condX3 <= 0
 c1680: - 2 aux_cap_helper__b1,r0,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b1,r0,band_HIGH_PAIRED__condX2 <= 0
 c1681: aux_cap_helper__b5,r9,band_LOW_PAIRED__Z2
        + aux_cap_helper__b5,r9,band_LOW_PAIRED__Z1  = 1
 c1682: sv_function_helpervar_b5,r10__condX3
        - 15.9946187432626 sv_function_helpervar_b5,r10__Z3 <= 0
 c1683: sv_function_helpervar_b9,r5__condX2
        - 7.61006379523071 sv_function_helpervar_b9,r5__Z2 <= 0
 c1684: - X__b7,r13,band_HIGH_PAIRED
        + aux_cap_helper__b7,r13,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b7,r13,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b7,r13,band_HIGH_PAIRED__condX1  = 0
 c1685: - aux_cap_b2,r10,band_HIGH_PAIRED
        + 1.3908260184 aux_cap_helper__b2,r10,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b2,r10,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b2,r10,band_HIGH_PAIRED__condX1
        - 0.6181448971 aux_cap_helper__b2,r10,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b2,r10,band_HIGH_PAIRED__Z3  = 0
 c1686: aux_cap_helper__b4,r0,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b4,r0,band_HIGH_PAIRED__Z2 <= 0
 c1687: aux_cap_helper__b6,r3,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b6,r3,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b6,r3,band_HIGH_PAIRED__Z1  = 1
 c1688: sv_function_helpervar_b4,r12__condX2
        + sv_function_helpervar_b4,r12__condX3
        + sv_function_helpervar_b4,r12__condX1 - aux_c_b4,r12  = 0
 c1689: aux_cap_helper__b8,r1,band_UNPAIRED__condX1
        + aux_cap_helper__b8,r1,band_UNPAIRED__condX2 - X__b8,r1,band_UNPAIRED
         = 0
 c1690: - aux_cap_helper__b8,r2,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b8,r2,band_HIGH_PAIRED__condX1 <= 0
 c1691: - aux_cap_helper__b6,r1,band_UNPAIRED__Z1
        + aux_cap_helper__b6,r1,band_UNPAIRED__condX1 <= 0
 c1692: - aux_cap_helper__b2,r7,band_UNPAIRED__Z1
        + aux_cap_helper__b2,r7,band_UNPAIRED__condX1 <= 0
 c1693: - 0.3214355447 aux_cap_helper__b8,r2,band_UNPAIRED__Z2
        - aux_cap_b8,r2,band_UNPAIRED
        + 1.1250244065 aux_cap_helper__b8,r2,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b8,r2,band_UNPAIRED__condX1  = 0
 c1694: aux_cap_b3,r4,band_LOW_PAIRED + aux_cap_b3,r4,band_UNPAIRED
        - aux_c_b3,r4 + aux_cap_b3,r4,band_HIGH_PAIRED  = 0
 c1695: - aux_Omega_b1,r9 + 417.695148939744 aux_sv_b1,r9  = 0
 c1696: aux_cap_helper__b1,r2,band_UNPAIRED__Z1
        + aux_cap_helper__b1,r2,band_UNPAIRED__Z2  = 1
 c1697: aux_cap_helper__b5,r7,band_UNPAIRED__Z1
        + aux_cap_helper__b5,r7,band_UNPAIRED__Z2  = 1
 c1698: - 0.159946187454999 sv_function_helpervar_b7,r9__Z1
        + sv_function_helpervar_b7,r9__condX1 <= 0
 c1699: sv_function_helpervar_b2,r8__condX2
        - 6.53881311386621 sv_function_helpervar_b2,r8__Z2 <= 0
 c1700: aux_cap_helper__b4,r3,band_UNPAIRED__Z1
        + aux_cap_helper__b4,r3,band_UNPAIRED__Z2  = 1
 c1701: aux_cap_helper__b0,r7,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b0,r7,band_HIGH_PAIRED__Z3 <= 0
 c1702: - sv_function_helpervar_b1,r11__condX2
        + 0.159946187471509 sv_function_helpervar_b1,r11__Z2 <= 0
 c1703: aux_cap_helper__b8,r8,band_UNPAIRED__Z1
        + aux_cap_helper__b8,r8,band_UNPAIRED__Z2  = 1
 c1704: - aux_cap_helper__b5,r13,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b5,r13,band_HIGH_PAIRED__condX1 <= 0
 c1705: aux_cap_helper__b0,r1,band_UNPAIRED__Z2
        - aux_cap_helper__b0,r1,band_UNPAIRED__condX2 <= 0
 c1706: aux_cap_helper__b7,r4,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b7,r4,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b7,r4,band_HIGH_PAIRED__Z3  = 1
 c1707: - 2 aux_cap_helper__b3,r9,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b3,r9,band_HIGH_PAIRED__condX2 <= 0
 c1708: sv_function_helpervar_b4,r13__condX1
        - 0.159946187484425 sv_function_helpervar_b4,r13__Z1 <= 0
 c1709: - sv_function_helpervar_b9,r3__condX3
        + 8.11597258834619 sv_function_helpervar_b9,r3__Z3 <= 0
 c1710: - 2 aux_cap_helper__b0,r12,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b0,r12,band_HIGH_PAIRED__condX2 <= 0
 c1711: - aux_cap_helper__b1,r6,band_HIGH_PAIRED__condX1 <= 0
 c1712: sv_function_helpervar_b7,r11__condX2
        - 6.9028137273107 sv_function_helpervar_b7,r11__Z2 <= 0
 c1713: - aux_cap_b0,r3,band_UNPAIRED
        - 0.3214355447 aux_cap_helper__b0,r3,band_UNPAIRED__Z2
        + 1.1250244065 aux_cap_helper__b0,r3,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b0,r3,band_UNPAIRED__condX1  = 0
 c1714: - 14 GlobalBidder_hat{W}__i8,k1 + GlobalBidder_W_i_8 >= -1
 c1715: aux_cap_helper__b3,r6,band_UNPAIRED__condX1
        - aux_cap_helper__b3,r6,band_UNPAIRED__Z1 <= 0
 c1716: - aux_cap_helper__b7,r6,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b7,r6,band_HIGH_PAIRED__Z2 <= 0
 c1717: aux_cap_helper__b8,r12,band_UNPAIRED__Z2
        - aux_cap_helper__b8,r12,band_UNPAIRED__condX2 <= 0
 c1718: - 2 aux_cap_helper__b8,r4,band_LOW_PAIRED__Z2
        + aux_cap_helper__b8,r4,band_LOW_PAIRED__condX2 <= 0
 c1719: - aux_Omega_b6,r2 + 103.278540175537 aux_sv_b6,r2  = 0
 c1720: aux_cap_helper__b5,r10,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b5,r10,band_LOW_PAIRED__Z2 <= 0
 c1721: - aux_cap_helper__b9,r9,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b9,r9,band_HIGH_PAIRED__Z3 <= 0
 c1722: - aux_cap_helper__b9,r3,band_LOW_PAIRED__condX1 <= 0
 c1723: - aux_cap_helper__b6,r11,band_UNPAIRED__condX1 <= 0
 c1724: - sv_function_helpervar_b5,r1__condX3
        + 5.85899563710341 sv_function_helpervar_b5,r1__Z3 <= 0
 c1725: - aux_cap_helper__b7,r6,band_HIGH_PAIRED__condX1 <= 0
 c1726: - aux_cap_helper__b4,r6,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b4,r6,band_HIGH_PAIRED__Z2 <= 0
 c1727: aux_cap_helper__b8,r4,band_LOW_PAIRED__condX1
        + aux_cap_helper__b8,r4,band_LOW_PAIRED__condX2
        - X__b8,r4,band_LOW_PAIRED  = 0
 c1728: aux_cap_helper__b1,r5,band_LOW_PAIRED__condX1
        - aux_cap_helper__b1,r5,band_LOW_PAIRED__Z1 <= 0
 c1729: - aux_c_b7,r10 + sv_function_helpervar_b7,r10__condX3
        + sv_function_helpervar_b7,r10__condX1
        + sv_function_helpervar_b7,r10__condX2  = 0
 c1730: - aux_cap_helper__b8,r12,band_UNPAIRED__condX1 <= 0
 c1731: aux_cap_helper__b0,r0,band_LOW_PAIRED__Z2
        - aux_cap_helper__b0,r0,band_LOW_PAIRED__condX2 <= 0
 c1732: 2 aux_cap_helper__b6,r9,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b6,r9,band_HIGH_PAIRED__condX3 <= 0
 c1733: 0.0102040816326531 X__b8,r1,band_LOW_PAIRED
        + 0.0102040816326531 X__b8,r1,band_HIGH_PAIRED
        + 0.0102040816326531 X__b8,r1,band_UNPAIRED - GlobalBidder_W_ir__b8,r1
        <= 0
 c1734: - 0.159946187438242 sv_function_helpervar_b0,r11__Z1
        + sv_function_helpervar_b0,r11__condX1 <= 0
 c1735: 5.97379865648078 sv_function_helpervar_b4,r0__Z3
        - sv_function_helpervar_b4,r0__condX3 <= 0
 c1736: - aux_cap_helper__b4,r6,band_HIGH_PAIRED__condX1 <= 0
 c1737: aux_cap_helper__b5,r0,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b5,r0,band_LOW_PAIRED__Z2 <= 0
 c1738: aux_cap_helper__b5,r8,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b5,r8,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b5,r8,band_HIGH_PAIRED__condX1
        - X__b5,r8,band_HIGH_PAIRED  = 0
 c1739: 2 aux_cap_helper__b7,r0,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b7,r0,band_HIGH_PAIRED__condX3 <= 0
 c1740: - sv_function_helpervar_b6,r2__condX1 <= 0
 c1741: aux_cap_helper__b9,r11,band_LOW_PAIRED__Z2
        + aux_cap_helper__b9,r11,band_LOW_PAIRED__Z1  = 1
 c1742: - aux_cap_helper__b1,r6,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b1,r6,band_HIGH_PAIRED__Z2 <= 0
 c1743: - aux_cap_helper__b0,r0,band_LOW_PAIRED__condX1 <= 0
 c1744: - sv_function_helpervar_b2,r10__condX3
        + 6.55138684469449 sv_function_helpervar_b2,r10__Z3 <= 0
 c1745: aux_cap_helper__b4,r10,band_UNPAIRED__Z2
        + aux_cap_helper__b4,r10,band_UNPAIRED__Z1  = 1
 c1746: aux_cap_helper__b2,r1,band_LOW_PAIRED__condX2
        + aux_cap_helper__b2,r1,band_LOW_PAIRED__condX1
        - X__b2,r1,band_LOW_PAIRED  = 0
 c1747: sv_function_helpervar_b1,r0__condX2
        + sv_function_helpervar_b1,r0__condX3
        + sv_function_helpervar_b1,r0__condX1 - aux_c_b1,r0  = 0
 c1748: - X__b5,r2,band_UNPAIRED + aux_cap_helper__b5,r2,band_UNPAIRED__condX1
        + aux_cap_helper__b5,r2,band_UNPAIRED__condX2  = 0
 c1749: 6.53655273965809 sv_function_helpervar_b7,r4__Z3
        - sv_function_helpervar_b7,r4__condX3 <= 0
 c1750: X__b9,r11,band_UNPAIRED - GlobalBidder_W_ir__b9,r11
        + X__b9,r11,band_HIGH_PAIRED + X__b9,r11,band_LOW_PAIRED >= 0
 c1751: - aux_cap_helper__b0,r8,band_LOW_PAIRED__condX2
        + aux_cap_helper__b0,r8,band_LOW_PAIRED__Z2 <= 0
 c1752: sv_function_helpervar_b4,r8__condX3
        - 15.9946187432626 sv_function_helpervar_b4,r8__Z3 <= 0
 c1753: 1101.78639494142 aux_sv_b8,r10 - aux_Omega_b8,r10  = 0
 c1754: - aux_cap_helper__b8,r1,band_LOW_PAIRED__condX2
        + aux_cap_helper__b8,r1,band_LOW_PAIRED__Z2 <= 0
 c1755: - aux_cap_helper__b3,r11,band_UNPAIRED__Z1
        + aux_cap_helper__b3,r11,band_UNPAIRED__condX1 <= 0
 c1756: aux_cap_helper__b9,r12,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b9,r12,band_LOW_PAIRED__Z2 <= 0
 c1757: - aux_cap_helper__b1,r10,band_UNPAIRED__condX1 <= 0
 c1758: - aux_cap_helper__b2,r0,band_UNPAIRED__condX2
        + aux_cap_helper__b2,r0,band_UNPAIRED__Z2 <= 0
 c1759: - aux_cap_helper__b0,r1,band_UNPAIRED__condX1 <= 0
 c1760: aux_cap_helper__b3,r3,band_UNPAIRED__condX1
        - aux_cap_helper__b3,r3,band_UNPAIRED__Z1 <= 0
 c1761: 6.40785831161345 sv_function_helpervar_b9,r9__Z3
        - sv_function_helpervar_b9,r9__condX3 <= 0
 c1762: - 0.6181448971 aux_cap_helper__b7,r11,band_HIGH_PAIRED__Z2
        + 1.3908260184 aux_cap_helper__b7,r11,band_HIGH_PAIRED__condX3
        + 1.5453622427 aux_cap_helper__b7,r11,band_HIGH_PAIRED__condX1
        + 2.1635071398 aux_cap_helper__b7,r11,band_HIGH_PAIRED__condX2
        - aux_cap_b7,r11,band_HIGH_PAIRED
        + 0.927217345653617 aux_cap_helper__b7,r11,band_HIGH_PAIRED__Z3  = 0
 c1763: aux_cap_helper__b3,r4,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b3,r4,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b3,r4,band_HIGH_PAIRED__Z3  = 1
 c1764: - aux_cap_helper__b1,r13,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b1,r13,band_HIGH_PAIRED__Z2 <= 0
 c1765: - 1.4943850124 aux_cap_helper__b8,r10,band_LOW_PAIRED__Z2
        + 3.735962531 aux_cap_helper__b8,r10,band_LOW_PAIRED__condX1
        + 5.2303475434 aux_cap_helper__b8,r10,band_LOW_PAIRED__condX2
        - aux_cap_b8,r10,band_LOW_PAIRED  = 0
 c1766: GlobalBidder_PSI__i9,k2 - 18497526.8940961 GlobalBidder_hat{W}__i9,k2
        <= 0
 c1767: 5.3036405285 sv_function_helpervar_b1,r7__condX2
        + 125.5635947333 sv_function_helpervar_b1,r7__condX1
        + 2.1403549978 sv_function_helpervar_b1,r7__condX3 - aux_sv_b1,r7
        + 19.2351211767582 sv_function_helpervar_b1,r7__Z2
        + 40.148868424629 sv_function_helpervar_b1,r7__Z3  = 0
 c1768: - X__b8,r10,band_LOW_PAIRED
        + aux_cap_helper__b8,r10,band_LOW_PAIRED__condX1
        + aux_cap_helper__b8,r10,band_LOW_PAIRED__condX2  = 0
 c1769: 6572.78131697886 aux_sv_b8,r7 - aux_Omega_b8,r7  = 0
 c1770: - 2 aux_cap_helper__b2,r10,band_LOW_PAIRED__Z2
        + aux_cap_helper__b2,r10,band_LOW_PAIRED__condX2 <= 0
 c1771: 4021.04411604543 aux_sv_b8,r11 - aux_Omega_b8,r11  = 0
 c1772: 1.1250244065 aux_cap_helper__b5,r13,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b5,r13,band_UNPAIRED__condX1
        - 0.3214355447 aux_cap_helper__b5,r13,band_UNPAIRED__Z2
        - aux_cap_b5,r13,band_UNPAIRED  = 0
 c1773: aux_cap_helper__b9,r6,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b9,r6,band_UNPAIRED__Z2 <= 0
 c1774: aux_cap_helper__b0,r8,band_UNPAIRED__Z2
        + aux_cap_helper__b0,r8,band_UNPAIRED__Z1  = 1
 c1775: - X__b7,r7,band_LOW_PAIRED
        + aux_cap_helper__b7,r7,band_LOW_PAIRED__condX2
        + aux_cap_helper__b7,r7,band_LOW_PAIRED__condX1  = 0
 c1776: aux_Omega_b9,r1 + aux_Omega_b9,r0 + aux_Omega_b9,r3 + aux_Omega_b9,r2
        + aux_Omega_b9,r5 + aux_Omega_b9,r4 + aux_Omega_b9,r7 + aux_Omega_b9,r6
        + aux_Omega_b9,r9 + aux_Omega_b9,r8 - GlobalBidder_PSI__i9,k0
        + aux_Omega_b9,r13 + aux_Omega_b9,r10 + aux_Omega_b9,r11
        + aux_Omega_b9,r12 + 18497526.8940961 GlobalBidder_hat{W}__i9,k0
        <= 18497526.8940961
 c1777: 2 aux_cap_helper__b8,r12,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b8,r12,band_HIGH_PAIRED__condX3 <= 0
 c1778: - aux_cap_helper__b1,r13,band_HIGH_PAIRED__condX1 <= 0
 c1779: - aux_cap_helper__b3,r4,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b3,r4,band_HIGH_PAIRED__condX1 <= 0
 c1780: - aux_cap_helper__b4,r13,band_UNPAIRED__condX1 <= 0
 c1781: - 15.9946187432626 sv_function_helpervar_b3,r5__Z3
        + sv_function_helpervar_b3,r5__condX3 <= 0
 c1782: aux_cap_helper__b0,r4,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b0,r4,band_HIGH_PAIRED__Z1 <= 0
 c1783: 3.735962531 aux_cap_helper__b1,r5,band_LOW_PAIRED__condX1
        + 5.2303475434 aux_cap_helper__b1,r5,band_LOW_PAIRED__condX2
        - aux_cap_b1,r5,band_LOW_PAIRED
        - 1.4943850124 aux_cap_helper__b1,r5,band_LOW_PAIRED__Z2  = 0
 c1784: 6.17377049158038 sv_function_helpervar_b4,r7__Z3
        - sv_function_helpervar_b4,r7__condX3 <= 0
 c1785: - aux_cap_helper__b8,r0,band_UNPAIRED__Z1
        + aux_cap_helper__b8,r0,band_UNPAIRED__condX1 <= 0
 c1786: aux_cap_b8,r4,band_HIGH_PAIRED - aux_c_b8,r4
        + aux_cap_b8,r4,band_LOW_PAIRED + aux_cap_b8,r4,band_UNPAIRED  = 0
 c1787: - aux_Omega_b2,r5 + 380.14505732379 aux_sv_b2,r5  = 0
 c1788: - 0.6181448971 aux_cap_helper__b5,r2,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b5,r2,band_HIGH_PAIRED__Z3
        + 1.3908260184 aux_cap_helper__b5,r2,band_HIGH_PAIRED__condX3
        + 1.5453622427 aux_cap_helper__b5,r2,band_HIGH_PAIRED__condX1
        + 2.1635071398 aux_cap_helper__b5,r2,band_HIGH_PAIRED__condX2
        - aux_cap_b5,r2,band_HIGH_PAIRED  = 0
 c1789: 7.03795147765963 sv_function_helpervar_b8,r1__Z3
        - sv_function_helpervar_b8,r1__condX3 <= 0
 c1790: aux_cap_helper__b3,r12,band_UNPAIRED__Z1
        + aux_cap_helper__b3,r12,band_UNPAIRED__Z2  = 1
 c1791: aux_cap_helper__b7,r12,band_UNPAIRED__condX1
        + aux_cap_helper__b7,r12,band_UNPAIRED__condX2
        - X__b7,r12,band_UNPAIRED  = 0
 c1792: 1.1250244065 aux_cap_helper__b7,r13,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b7,r13,band_UNPAIRED__condX1
        - 0.3214355447 aux_cap_helper__b7,r13,band_UNPAIRED__Z2
        - aux_cap_b7,r13,band_UNPAIRED  = 0
 c1793: 1.3908260184 aux_cap_helper__b3,r11,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b3,r11,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b3,r11,band_HIGH_PAIRED__condX1
        - aux_cap_b3,r11,band_HIGH_PAIRED
        - 0.6181448971 aux_cap_helper__b3,r11,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b3,r11,band_HIGH_PAIRED__Z3  = 0
 c1794: 1.5453622427 aux_cap_helper__b4,r6,band_HIGH_PAIRED__condX1
        + 1.3908260184 aux_cap_helper__b4,r6,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b4,r6,band_HIGH_PAIRED__condX2
        - aux_cap_b4,r6,band_HIGH_PAIRED
        - 0.6181448971 aux_cap_helper__b4,r6,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b4,r6,band_HIGH_PAIRED__Z3  = 0
 c1795: - aux_cap_helper__b0,r8,band_LOW_PAIRED__condX1 <= 0
 c1796: sv_function_helpervar_b7,r4__Z1 + sv_function_helpervar_b7,r4__Z2
        + sv_function_helpervar_b7,r4__Z3  = 1
 c1797: - 0.6181448971 aux_cap_helper__b8,r12,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b8,r12,band_HIGH_PAIRED__Z3
        - aux_cap_b8,r12,band_HIGH_PAIRED
        + 1.3908260184 aux_cap_helper__b8,r12,band_HIGH_PAIRED__condX3
        + 1.5453622427 aux_cap_helper__b8,r12,band_HIGH_PAIRED__condX1
        + 2.1635071398 aux_cap_helper__b8,r12,band_HIGH_PAIRED__condX2  = 0
 c1798: - aux_cap_helper__b3,r9,band_LOW_PAIRED__condX1 <= 0
 c1799: - 5.78594863013166 sv_function_helpervar_b3,r12__Z2
        + sv_function_helpervar_b3,r12__condX2 <= 0
 c1800: sv_function_helpervar_b5,r9__condX2
        - 6.12746312836256 sv_function_helpervar_b5,r9__Z2 <= 0
 c1801: sv_function_helpervar_b8,r2__condX3
        + sv_function_helpervar_b8,r2__condX2
        + sv_function_helpervar_b8,r2__condX1 - aux_c_b8,r2  = 0
 c1802: 6.12210391322581 sv_function_helpervar_b0,r7__Z3
        - sv_function_helpervar_b0,r7__condX3 <= 0
 c1803: aux_cap_helper__b1,r4,band_UNPAIRED__Z2
        - aux_cap_helper__b1,r4,band_UNPAIRED__condX2 <= 0
 c1804: aux_cap_helper__b6,r13,band_LOW_PAIRED__condX1
        + aux_cap_helper__b6,r13,band_LOW_PAIRED__condX2
        - X__b6,r13,band_LOW_PAIRED  = 0
 c1805: - aux_cap_helper__b2,r0,band_UNPAIRED__condX1 <= 0
 c1806: aux_cap_helper__b7,r11,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b7,r11,band_LOW_PAIRED__Z2 <= 0
 c1807: X__b8,r0,band_LOW_PAIRED + X__b8,r0,band_HIGH_PAIRED
        + X__b8,r0,band_UNPAIRED - GlobalBidder_W_ir__b8,r0 >= 0
 c1808: aux_cap_helper__b1,r4,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b1,r4,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b1,r4,band_HIGH_PAIRED__Z3  = 1
 c1809: - aux_cap_b6,r1,band_LOW_PAIRED
        - 1.4943850124 aux_cap_helper__b6,r1,band_LOW_PAIRED__Z2
        + 5.2303475434 aux_cap_helper__b6,r1,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b6,r1,band_LOW_PAIRED__condX1  = 0
 c1810: sv_function_helpervar_b1,r2__condX1 - aux_c_b1,r2
        + sv_function_helpervar_b1,r2__condX3
        + sv_function_helpervar_b1,r2__condX2  = 0
 c1811: 1.1250244065 aux_cap_helper__b8,r4,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b8,r4,band_UNPAIRED__condX1
        - aux_cap_b8,r4,band_UNPAIRED
        - 0.3214355447 aux_cap_helper__b8,r4,band_UNPAIRED__Z2  = 0
 c1812: 1059.4718674594 sv_function_helpervar_b6,r13__condX1
        + 53.7997177554 sv_function_helpervar_b6,r13__condX2
        + 16.187709585 sv_function_helpervar_b6,r13__condX3
        + 368.707778759736 sv_function_helpervar_b6,r13__Z3
        + 160.853426171415 sv_function_helpervar_b6,r13__Z2 - aux_sv_b6,r13
         = 0
 c1813: - aux_cap_helper__b1,r4,band_UNPAIRED__condX1 <= 0
 c1814: - 2 aux_cap_helper__b4,r11,band_LOW_PAIRED__Z2
        + aux_cap_helper__b4,r11,band_LOW_PAIRED__condX2 <= 0
 c1815: - aux_cap_helper__b4,r12,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b4,r12,band_HIGH_PAIRED__condX1 <= 0
 c1816: 5.2303475434 aux_cap_helper__b7,r11,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b7,r11,band_LOW_PAIRED__condX1
        - aux_cap_b7,r11,band_LOW_PAIRED
        - 1.4943850124 aux_cap_helper__b7,r11,band_LOW_PAIRED__Z2  = 0
 c1817: sv_function_helpervar_b3,r13__condX1
        - 0.159946187447925 sv_function_helpervar_b3,r13__Z1 <= 0
 c1818: - sv_function_helpervar_b3,r13__condX3
        + 5.70920988297092 sv_function_helpervar_b3,r13__Z3 <= 0
 c1819: - 2 aux_cap_helper__b4,r3,band_UNPAIRED__Z2
        + aux_cap_helper__b4,r3,band_UNPAIRED__condX2 <= 0
 c1820: aux_cap_helper__b2,r6,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b2,r6,band_UNPAIRED__Z2 <= 0
 c1821: - sv_function_helpervar_b3,r7__condX2
        + 0.159946187433093 sv_function_helpervar_b3,r7__Z2 <= 0
 c1822: 0.159946187436079 sv_function_helpervar_b5,r7__Z2
        - sv_function_helpervar_b5,r7__condX2 <= 0
 c1823: - sv_function_helpervar_b0,r5__condX2
        + 0.159946187439545 sv_function_helpervar_b0,r5__Z2 <= 0
 c1824: aux_cap_helper__b9,r2,band_UNPAIRED__Z2
        - aux_cap_helper__b9,r2,band_UNPAIRED__condX2 <= 0
 c1825: - sv_function_helpervar_b7,r3__condX3
        + 7.00344924003967 sv_function_helpervar_b7,r3__Z3 <= 0
 c1826: - aux_cap_helper__b6,r13,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b6,r13,band_HIGH_PAIRED__Z3 <= 0
 c1827: - aux_cap_helper__b4,r5,band_LOW_PAIRED__condX1 <= 0
 c1828: - aux_cap_b4,r11,band_UNPAIRED
        - 0.3214355447 aux_cap_helper__b4,r11,band_UNPAIRED__Z2
        + 1.1250244065 aux_cap_helper__b4,r11,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b4,r11,band_UNPAIRED__condX1  = 0
 c1829: aux_cap_helper__b1,r11,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b1,r11,band_LOW_PAIRED__Z2 <= 0
 c1830: aux_cap_helper__b0,r2,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b0,r2,band_HIGH_PAIRED__Z2 <= 0
 c1831: - 2 aux_cap_helper__b1,r1,band_UNPAIRED__Z2
        + aux_cap_helper__b1,r1,band_UNPAIRED__condX2 <= 0
 c1832: aux_cap_helper__b3,r6,band_LOW_PAIRED__Z2
        - aux_cap_helper__b3,r6,band_LOW_PAIRED__condX2 <= 0
 c1833: - X__b6,r6,band_UNPAIRED + aux_cap_helper__b6,r6,band_UNPAIRED__condX1
        + aux_cap_helper__b6,r6,band_UNPAIRED__condX2  = 0
 c1834: aux_cap_helper__b6,r0,band_UNPAIRED__Z2
        - aux_cap_helper__b6,r0,band_UNPAIRED__condX2 <= 0
 c1835: aux_cap_helper__b7,r11,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b7,r11,band_HIGH_PAIRED__Z3 <= 0
 c1836: aux_cap_helper__b3,r2,band_LOW_PAIRED__Z1
        + aux_cap_helper__b3,r2,band_LOW_PAIRED__Z2  = 1
 c1837: aux_cap_helper__b4,r11,band_UNPAIRED__Z2
        - aux_cap_helper__b4,r11,band_UNPAIRED__condX2 <= 0
 c1838: aux_cap_helper__b4,r12,band_LOW_PAIRED__Z2
        - aux_cap_helper__b4,r12,band_LOW_PAIRED__condX2 <= 0
 c1839: - aux_cap_helper__b5,r4,band_LOW_PAIRED__condX1 <= 0
 c1840: - aux_cap_helper__b3,r6,band_LOW_PAIRED__condX1 <= 0
 c1841: 2 aux_cap_helper__b2,r8,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b2,r8,band_HIGH_PAIRED__condX3 <= 0
 c1842: - aux_cap_helper__b4,r5,band_LOW_PAIRED__condX2
        + aux_cap_helper__b4,r5,band_LOW_PAIRED__Z2 <= 0
 c1843: 1719.09458806 sv_function_helpervar_b9,r5__condX1
        + 62.8788520737 sv_function_helpervar_b9,r5__condX2
        + 32.7939439765 sv_function_helpervar_b9,r5__condX3
        + 493.853462466146 sv_function_helpervar_b9,r5__Z3
        + 264.905392572811 sv_function_helpervar_b9,r5__Z2 - aux_sv_b9,r5  = 0
 c1844: 5.54618304423859 sv_function_helpervar_b5,r12__Z3
        - sv_function_helpervar_b5,r12__condX3 <= 0
 c1845: aux_cap_helper__b2,r7,band_LOW_PAIRED__condX1
        - aux_cap_helper__b2,r7,band_LOW_PAIRED__Z1 <= 0
 c1846: - X__b4,r3,band_HIGH_PAIRED
        + aux_cap_helper__b4,r3,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b4,r3,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b4,r3,band_HIGH_PAIRED__condX1  = 0
 c1847: - aux_cap_helper__b4,r12,band_LOW_PAIRED__condX1 <= 0
 c1848: - 2 aux_cap_helper__b3,r12,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b3,r12,band_HIGH_PAIRED__condX2 <= 0
 c1849: 0.0102040816326531 X__b8,r11,band_LOW_PAIRED
        + 0.0102040816326531 X__b8,r11,band_HIGH_PAIRED
        - GlobalBidder_W_ir__b8,r11
        + 0.0102040816326531 X__b8,r11,band_UNPAIRED <= 0
 c1850: - 2 aux_cap_helper__b7,r5,band_UNPAIRED__Z2
        + aux_cap_helper__b7,r5,band_UNPAIRED__condX2 <= 0
 c1851: - aux_cap_b1,r3,band_UNPAIRED
        + 1.1250244065 aux_cap_helper__b1,r3,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b1,r3,band_UNPAIRED__condX1
        - 0.3214355447 aux_cap_helper__b1,r3,band_UNPAIRED__Z2  = 0
 c1852: - aux_cap_b4,r5,band_UNPAIRED
        - 0.3214355447 aux_cap_helper__b4,r5,band_UNPAIRED__Z2
        + 1.1250244065 aux_cap_helper__b4,r5,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b4,r5,band_UNPAIRED__condX1  = 0
 c1853: 1.1250244065 aux_cap_helper__b7,r7,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b7,r7,band_UNPAIRED__condX1
        - 0.3214355447 aux_cap_helper__b7,r7,band_UNPAIRED__Z2
        - aux_cap_b7,r7,band_UNPAIRED  = 0
 c1854: aux_cap_helper__b7,r13,band_LOW_PAIRED__condX1
        - aux_cap_helper__b7,r13,band_LOW_PAIRED__Z1 <= 0
 c1855: - sv_function_helpervar_b5,r10__condX1 <= 0
 c1856: - aux_cap_helper__b3,r13,band_LOW_PAIRED__condX1 <= 0
 c1857: sv_function_helpervar_b6,r6__condX1
        - 0.159946187491039 sv_function_helpervar_b6,r6__Z1 <= 0
 c1858: sv_function_helpervar_b9,r5__condX1
        + sv_function_helpervar_b9,r5__condX2
        + sv_function_helpervar_b9,r5__condX3 - aux_c_b9,r5  = 0
 c1859: - aux_c_b5,r12 + sv_function_helpervar_b5,r12__condX3
        + sv_function_helpervar_b5,r12__condX2
        + sv_function_helpervar_b5,r12__condX1  = 0
 c1860: - sv_function_helpervar_b9,r10__condX1 <= 0
 c1861: aux_cap_helper__b7,r13,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b7,r13,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b7,r13,band_HIGH_PAIRED__Z3  = 1
 c1862: 0.159946187436557 sv_function_helpervar_b6,r0__Z2
        - sv_function_helpervar_b6,r0__condX2 <= 0
 c1863: - X__b1,r7,band_HIGH_PAIRED
        + aux_cap_helper__b1,r7,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b1,r7,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b1,r7,band_HIGH_PAIRED__condX1  = 0
 c1864: - aux_cap_helper__b9,r2,band_UNPAIRED__condX1 <= 0
 c1865: aux_cap_helper__b2,r0,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b2,r0,band_LOW_PAIRED__Z2 <= 0
 c1866: aux_cap_helper__b6,r3,band_LOW_PAIRED__Z2
        - aux_cap_helper__b6,r3,band_LOW_PAIRED__condX2 <= 0
 c1867: - aux_cap_helper__b8,r0,band_HIGH_PAIRED__condX1 <= 0
 c1868: aux_cap_helper__b0,r11,band_UNPAIRED__condX2
        + aux_cap_helper__b0,r11,band_UNPAIRED__condX1
        - X__b0,r11,band_UNPAIRED  = 0
 c1869: - aux_cap_helper__b1,r8,band_LOW_PAIRED__Z1
        + aux_cap_helper__b1,r8,band_LOW_PAIRED__condX1 <= 0
 c1870: - X__b9,r4,band_HIGH_PAIRED
        + aux_cap_helper__b9,r4,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b9,r4,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b9,r4,band_HIGH_PAIRED__condX3  = 0
 c1871: - aux_cap_helper__b6,r0,band_UNPAIRED__condX1 <= 0
 c1872: - aux_cap_helper__b9,r4,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b9,r4,band_HIGH_PAIRED__Z2 <= 0
 c1873: aux_cap_helper__b3,r8,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b3,r8,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b3,r8,band_HIGH_PAIRED__condX2
        - X__b3,r8,band_HIGH_PAIRED  = 0
 c1874: aux_cap_helper__b2,r2,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b2,r2,band_HIGH_PAIRED__Z3 <= 0
 c1875: - aux_cap_helper__b6,r3,band_LOW_PAIRED__condX1 <= 0
 c1876: sv_function_helpervar_b8,r8__condX3
        + sv_function_helpervar_b8,r8__condX2
        + sv_function_helpervar_b8,r8__condX1 - aux_c_b8,r8  = 0
 c1877: X__b0,r9,band_LOW_PAIRED + X__b9,r9,band_LOW_PAIRED
        + X__b2,r9,band_LOW_PAIRED + X__b6,r9,band_LOW_PAIRED
        + X__b7,r9,band_LOW_PAIRED + X__b5,r9,band_LOW_PAIRED
        + X__b8,r9,band_LOW_PAIRED + X__b4,r9,band_LOW_PAIRED
        + X__b1,r9,band_LOW_PAIRED + X__b3,r9,band_LOW_PAIRED <= 2
 c1878: - 2 aux_cap_helper__b0,r2,band_LOW_PAIRED__Z2
        + aux_cap_helper__b0,r2,band_LOW_PAIRED__condX2 <= 0
 c1879: aux_cap_helper__b0,r12,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b0,r12,band_LOW_PAIRED__Z2 <= 0
 c1880: aux_cap_helper__b4,r1,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b4,r1,band_HIGH_PAIRED__Z1 <= 0
 c1881: - aux_c_b5,r7 + sv_function_helpervar_b5,r7__condX1
        + sv_function_helpervar_b5,r7__condX2
        + sv_function_helpervar_b5,r7__condX3  = 0
 c1882: - aux_cap_helper__b7,r9,band_LOW_PAIRED__Z1
        + aux_cap_helper__b7,r9,band_LOW_PAIRED__condX1 <= 0
 c1883: aux_cap_helper__b4,r8,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b4,r8,band_HIGH_PAIRED__Z3 <= 0
 c1884: sv_function_helpervar_b9,r13__condX2
        - 8.31490900094671 sv_function_helpervar_b9,r13__Z2 <= 0
 c1885: aux_cap_helper__b3,r2,band_UNPAIRED__Z2
        + aux_cap_helper__b3,r2,band_UNPAIRED__Z1  = 1
 c1886: sv_function_helpervar_b3,r10__condX2
        - 5.97509000744268 sv_function_helpervar_b3,r10__Z2 <= 0
 c1887: 6.12424674439599 sv_function_helpervar_b5,r2__Z3
        - sv_function_helpervar_b5,r2__condX3 <= 0
 c1888: - aux_cap_helper__b9,r4,band_HIGH_PAIRED__condX1 <= 0
 c1889: - aux_cap_helper__b3,r13,band_LOW_PAIRED__condX2
        + aux_cap_helper__b3,r13,band_LOW_PAIRED__Z2 <= 0
 c1890: GlobalBidder_W_i_9 + 14 GlobalBidder_hat{W}__i9,k0 <= 28
 c1891: - aux_cap_helper__b4,r8,band_UNPAIRED__condX1 <= 0
 c1892: aux_cap_b2,r10,band_LOW_PAIRED + aux_cap_b2,r10,band_HIGH_PAIRED
        - aux_c_b2,r10 + aux_cap_b2,r10,band_UNPAIRED  = 0
 c1893: aux_cap_helper__b8,r1,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b8,r1,band_LOW_PAIRED__Z2 <= 0
 c1894: - aux_cap_helper__b0,r13,band_UNPAIRED__Z1
        + aux_cap_helper__b0,r13,band_UNPAIRED__condX1 <= 0
 c1895: sv_function_helpervar_b5,r2__Z3 + sv_function_helpervar_b5,r2__Z2
        + sv_function_helpervar_b5,r2__Z1  = 1
 c1896: X__b8,r6,band_HIGH_PAIRED + X__b8,r6,band_UNPAIRED
        - GlobalBidder_W_ir__b8,r6 + X__b8,r6,band_LOW_PAIRED >= 0
 c1897: - aux_cap_helper__b6,r7,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b6,r7,band_HIGH_PAIRED__condX1 <= 0
 c1898: - aux_cap_helper__b2,r7,band_HIGH_PAIRED__condX1 <= 0
 c1899: - aux_cap_helper__b8,r2,band_UNPAIRED__Z1
        + aux_cap_helper__b8,r2,band_UNPAIRED__condX1 <= 0
 c1900: 5.2303475434 aux_cap_helper__b0,r11,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b0,r11,band_LOW_PAIRED__condX1
        - 1.4943850124 aux_cap_helper__b0,r11,band_LOW_PAIRED__Z2
        - aux_cap_b0,r11,band_LOW_PAIRED  = 0
 c1901: aux_cap_helper__b1,r12,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b1,r12,band_UNPAIRED__Z2 <= 0
 c1902: - aux_cap_helper__b1,r6,band_UNPAIRED__condX1 <= 0
 c1903: aux_cap_helper__b1,r10,band_LOW_PAIRED__Z1
        + aux_cap_helper__b1,r10,band_LOW_PAIRED__Z2  = 1
 c1904: sv_function_helpervar_b7,r7__condX3
        - 15.9946187432626 sv_function_helpervar_b7,r7__Z3 <= 0
 c1905: - aux_cap_helper__b0,r3,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b0,r3,band_HIGH_PAIRED__Z3 <= 0
 c1906: - aux_cap_helper__b5,r11,band_LOW_PAIRED__condX1 <= 0
 c1907: - aux_cap_helper__b8,r8,band_LOW_PAIRED__Z1
        + aux_cap_helper__b8,r8,band_LOW_PAIRED__condX1 <= 0
 c1908: - 15.9946187432626 sv_function_helpervar_b3,r4__Z3
        + sv_function_helpervar_b3,r4__condX3 <= 0
 c1909: sv_function_helpervar_b8,r12__condX2
        - 8.09819079914792 sv_function_helpervar_b8,r12__Z2 <= 0
 c1910: - aux_cap_helper__b9,r3,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b9,r3,band_HIGH_PAIRED__Z3 <= 0
 c1911: - 5.65569045862011 sv_function_helpervar_b2,r5__Z2
        + sv_function_helpervar_b2,r5__condX2 <= 0
 c1912: - 0.159946187465026 sv_function_helpervar_b4,r7__Z1
        + sv_function_helpervar_b4,r7__condX1 <= 0
 c1913: - X__b8,r12,band_UNPAIRED
        + aux_cap_helper__b8,r12,band_UNPAIRED__condX2
        + aux_cap_helper__b8,r12,band_UNPAIRED__condX1  = 0
 c1914: sv_function_helpervar_b7,r7__Z1 + sv_function_helpervar_b7,r7__Z2
        + sv_function_helpervar_b7,r7__Z3  = 1
 c1915: - aux_cap_helper__b3,r11,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b3,r11,band_HIGH_PAIRED__Z2 <= 0
 c1916: aux_cap_helper__b8,r10,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b8,r10,band_HIGH_PAIRED__condX2 <= 0
 c1917: - 1.4943850124 aux_cap_helper__b8,r5,band_LOW_PAIRED__Z2
        + 5.2303475434 aux_cap_helper__b8,r5,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b8,r5,band_LOW_PAIRED__condX1
        - aux_cap_b8,r5,band_LOW_PAIRED  = 0
 c1918: aux_cap_helper__b6,r5,band_UNPAIRED__Z2
        - aux_cap_helper__b6,r5,band_UNPAIRED__condX2 <= 0
 c1919: aux_cap_helper__b8,r4,band_LOW_PAIRED__Z2
        - aux_cap_helper__b8,r4,band_LOW_PAIRED__condX2 <= 0
 c1920: - aux_cap_helper__b8,r10,band_HIGH_PAIRED__condX1 <= 0
 c1921: aux_cap_helper__b6,r11,band_LOW_PAIRED__Z1
        + aux_cap_helper__b6,r11,band_LOW_PAIRED__Z2  = 1
 c1922: - aux_c_b7,r1 + aux_cap_b7,r1,band_HIGH_PAIRED
        + aux_cap_b7,r1,band_LOW_PAIRED + aux_cap_b7,r1,band_UNPAIRED  = 0
 c1923: - aux_cap_helper__b3,r11,band_HIGH_PAIRED__condX1 <= 0
 c1924: - aux_cap_helper__b2,r7,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b2,r7,band_HIGH_PAIRED__Z2 <= 0
 c1925: 1.1250244065 aux_cap_helper__b7,r10,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b7,r10,band_UNPAIRED__condX1
        - aux_cap_b7,r10,band_UNPAIRED
        - 0.3214355447 aux_cap_helper__b7,r10,band_UNPAIRED__Z2  = 0
 c1926: 2 aux_cap_helper__b5,r5,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b5,r5,band_HIGH_PAIRED__condX3 <= 0
 c1927: GlobalBidder_W_i_9 + 14 GlobalBidder_hat{W}__i9,k2 <= 26
 c1928: - aux_cap_helper__b1,r6,band_UNPAIRED__condX2
        + aux_cap_helper__b1,r6,band_UNPAIRED__Z2 <= 0
 c1929: - aux_cap_helper__b6,r5,band_UNPAIRED__condX1 <= 0
 c1930: aux_cap_helper__b4,r5,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b4,r5,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b4,r5,band_HIGH_PAIRED__Z2  = 1
 c1931: aux_cap_helper__b5,r11,band_LOW_PAIRED__Z2
        - aux_cap_helper__b5,r11,band_LOW_PAIRED__condX2 <= 0
 c1932: aux_cap_helper__b9,r3,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b9,r3,band_LOW_PAIRED__Z2 <= 0
 c1933: - 5.746806695232 sv_function_helpervar_b0,r9__Z2
        + sv_function_helpervar_b0,r9__condX2 <= 0
 c1934: aux_cap_helper__b4,r11,band_UNPAIRED__Z2
        + aux_cap_helper__b4,r11,band_UNPAIRED__Z1  = 1
 c1935: 5.48061618636332 sv_function_helpervar_b5,r7__Z3
        - sv_function_helpervar_b5,r7__condX3 <= 0
 c1936: - sv_function_helpervar_b9,r0__condX2
        + 0.159946187463129 sv_function_helpervar_b9,r0__Z2 <= 0
 c1937: aux_cap_b4,r1,band_UNPAIRED + aux_cap_b4,r1,band_HIGH_PAIRED
        + aux_cap_b4,r1,band_LOW_PAIRED - aux_c_b4,r1  = 0
 c1938: aux_cap_helper__b6,r9,band_LOW_PAIRED__Z2
        + aux_cap_helper__b6,r9,band_LOW_PAIRED__Z1  = 1
 c1939: - 14 GlobalBidder_hat{W}__i8,k0 + GlobalBidder_W_i_8 >= 0
 c1940: - aux_Omega_b2,r2 + 184.323809079769 aux_sv_b2,r2  = 0
 c1941: - X__b2,r13,band_HIGH_PAIRED
        + aux_cap_helper__b2,r13,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b2,r13,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b2,r13,band_HIGH_PAIRED__condX1  = 0
 c1942: aux_cap_helper__b3,r13,band_UNPAIRED__Z2
        + aux_cap_helper__b3,r13,band_UNPAIRED__Z1  = 1
 c1943: - aux_sv_b8,r11 + 25.9463646899 sv_function_helpervar_b8,r11__condX3
        + 1343.0085749947 sv_function_helpervar_b8,r11__condX1
        + 48.4363858437 sv_function_helpervar_b8,r11__condX2
        + 207.061886100482 sv_function_helpervar_b8,r11__Z2
        + 380.587053281055 sv_function_helpervar_b8,r11__Z3  = 0
 c1944: aux_cap_helper__b0,r9,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b0,r9,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b0,r9,band_HIGH_PAIRED__Z3  = 1
 c1945: aux_cap_helper__b9,r11,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b9,r11,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b9,r11,band_HIGH_PAIRED__Z3  = 1
 c1946: aux_cap_helper__b9,r12,band_UNPAIRED__Z2
        + aux_cap_helper__b9,r12,band_UNPAIRED__Z1  = 1
 c1947: - 2 aux_cap_helper__b5,r4,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b5,r4,band_HIGH_PAIRED__condX2 <= 0
 c1948: aux_cap_helper__b6,r0,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b6,r0,band_HIGH_PAIRED__Z3 <= 0
 c1949: aux_cap_helper__b1,r3,band_UNPAIRED__condX2
        + aux_cap_helper__b1,r3,band_UNPAIRED__condX1 - X__b1,r3,band_UNPAIRED
         = 0
 c1950: aux_cap_helper__b2,r7,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b2,r7,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b2,r7,band_HIGH_PAIRED__Z3  = 1
 c1951: aux_cap_helper__b0,r11,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b0,r11,band_HIGH_PAIRED__Z1 <= 0
 c1952: aux_cap_helper__b8,r1,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b8,r1,band_HIGH_PAIRED__Z2 <= 0
 c1953: - X__b6,r5,band_LOW_PAIRED
        + aux_cap_helper__b6,r5,band_LOW_PAIRED__condX2
        + aux_cap_helper__b6,r5,band_LOW_PAIRED__condX1  = 0
 c1954: aux_cap_helper__b2,r1,band_UNPAIRED__Z2
        + aux_cap_helper__b2,r1,band_UNPAIRED__Z1  = 1
 c1955: - aux_sv_b8,r13 + 207.245809210254 sv_function_helpervar_b8,r13__Z2
        + 371.167686045112 sv_function_helpervar_b8,r13__Z3
        + 1343.0085755433 sv_function_helpervar_b8,r13__condX1
        + 47.2864796812 sv_function_helpervar_b8,r13__condX2
        + 26.5352732089 sv_function_helpervar_b8,r13__condX3  = 0
 c1956: - 2 aux_cap_helper__b7,r0,band_UNPAIRED__Z2
        + aux_cap_helper__b7,r0,band_UNPAIRED__condX2 <= 0
 c1957: 3.735962531 aux_cap_helper__b1,r9,band_LOW_PAIRED__condX1
        + 5.2303475434 aux_cap_helper__b1,r9,band_LOW_PAIRED__condX2
        - aux_cap_b1,r9,band_LOW_PAIRED
        - 1.4943850124 aux_cap_helper__b1,r9,band_LOW_PAIRED__Z2  = 0
 c1958: aux_cap_helper__b9,r0,band_LOW_PAIRED__Z1
        + aux_cap_helper__b9,r0,band_LOW_PAIRED__Z2  = 1
 c1959: - aux_cap_helper__b2,r4,band_LOW_PAIRED__condX1 <= 0
 c1960: - aux_cap_helper__b7,r2,band_LOW_PAIRED__condX2
        + aux_cap_helper__b7,r2,band_LOW_PAIRED__Z2 <= 0
 c1961: - aux_cap_helper__b5,r7,band_LOW_PAIRED__condX2
        + aux_cap_helper__b5,r7,band_LOW_PAIRED__Z2 <= 0
 c1962: aux_cap_helper__b4,r2,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b4,r2,band_HIGH_PAIRED__condX2 <= 0
 c1963: sv_function_helpervar_b1,r11__Z1 + sv_function_helpervar_b1,r11__Z3
        + sv_function_helpervar_b1,r11__Z2  = 1
 c1964: - 1.4943850124 aux_cap_helper__b6,r4,band_LOW_PAIRED__Z2
        - aux_cap_b6,r4,band_LOW_PAIRED
        + 3.735962531 aux_cap_helper__b6,r4,band_LOW_PAIRED__condX1
        + 5.2303475434 aux_cap_helper__b6,r4,band_LOW_PAIRED__condX2  = 0
 c1965: 2 aux_cap_helper__b9,r13,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b9,r13,band_HIGH_PAIRED__condX3 <= 0
 c1966: 6.09869560127261 sv_function_helpervar_b1,r0__Z3
        - sv_function_helpervar_b1,r0__condX3 <= 0
 c1967: aux_cap_helper__b7,r7,band_UNPAIRED__Z1
        + aux_cap_helper__b7,r7,band_UNPAIRED__Z2  = 1
 c1968: - 18497526.8940961 GlobalBidder_hat{W}__i8,k0 + GlobalBidder_PSI__i8,k0
        <= 0
 c1969: aux_cap_helper__b0,r9,band_UNPAIRED__Z2
        + aux_cap_helper__b0,r9,band_UNPAIRED__Z1  = 1
 c1970: - aux_cap_helper__b4,r11,band_UNPAIRED__condX1 <= 0
 c1971: sv_function_helpervar_b4,r3__condX1
        + sv_function_helpervar_b4,r3__condX3
        + sv_function_helpervar_b4,r3__condX2 - aux_c_b4,r3  = 0
 c1972: - sv_function_helpervar_b8,r5__condX1 <= 0
 c1973: aux_cap_b4,r2,band_UNPAIRED + aux_cap_b4,r2,band_HIGH_PAIRED
        + aux_cap_b4,r2,band_LOW_PAIRED - aux_c_b4,r2  = 0
 c1974: aux_cap_helper__b5,r8,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b5,r8,band_UNPAIRED__Z2 <= 0
 c1975: sv_function_helpervar_b8,r11__Z1 + sv_function_helpervar_b8,r11__Z2
        + sv_function_helpervar_b8,r11__Z3  = 1
 c1976: aux_cap_helper__b4,r11,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b4,r11,band_HIGH_PAIRED__Z3 <= 0
 c1977: - aux_cap_helper__b4,r2,band_HIGH_PAIRED__condX1 <= 0
 c1978: - aux_cap_helper__b7,r2,band_LOW_PAIRED__condX1 <= 0
 c1979: aux_cap_helper__b0,r9,band_LOW_PAIRED__condX1
        - aux_cap_helper__b0,r9,band_LOW_PAIRED__Z1 <= 0
 c1980: aux_cap_helper__b7,r10,band_UNPAIRED__condX1
        - aux_cap_helper__b7,r10,band_UNPAIRED__Z1 <= 0
 c1981: aux_cap_helper__b1,r5,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b1,r5,band_HIGH_PAIRED__Z1 <= 0
 c1982: aux_cap_helper__b2,r11,band_UNPAIRED__condX1
        - aux_cap_helper__b2,r11,band_UNPAIRED__Z1 <= 0
 c1983: aux_cap_helper__b7,r10,band_LOW_PAIRED__condX1
        - aux_cap_helper__b7,r10,band_LOW_PAIRED__Z1 <= 0
 c1984: sv_function_helpervar_b7,r11__condX1
        - 0.159946187437288 sv_function_helpervar_b7,r11__Z1 <= 0
 c1985: - sv_function_helpervar_b2,r10__condX2
        + 0.159946187444046 sv_function_helpervar_b2,r10__Z2 <= 0
 c1986: aux_cap_helper__b1,r0,band_UNPAIRED__Z2
        + aux_cap_helper__b1,r0,band_UNPAIRED__Z1  = 1
 c1987: - aux_cap_helper__b3,r3,band_LOW_PAIRED__condX1 <= 0
 c1988: - aux_cap_b1,r12,band_HIGH_PAIRED
        + 0.927217345653617 aux_cap_helper__b1,r12,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b1,r12,band_HIGH_PAIRED__Z2
        + 1.3908260184 aux_cap_helper__b1,r12,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b1,r12,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b1,r12,band_HIGH_PAIRED__condX1  = 0
 c1989: - 6.22135306242807 sv_function_helpervar_b3,r5__Z2
        + sv_function_helpervar_b3,r5__condX2 <= 0
 c1990: - aux_cap_helper__b6,r11,band_HIGH_PAIRED__condX1 <= 0
 c1991: - 0.159946187462509 sv_function_helpervar_b8,r3__Z1
        + sv_function_helpervar_b8,r3__condX1 <= 0
 c1992: - aux_cap_helper__b6,r10,band_LOW_PAIRED__condX1 <= 0
 c1993: - 2 aux_cap_helper__b9,r2,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b9,r2,band_HIGH_PAIRED__condX2 <= 0
 c1994: aux_cap_helper__b2,r4,band_LOW_PAIRED__Z2
        - aux_cap_helper__b2,r4,band_LOW_PAIRED__condX2 <= 0
 c1995: 18497526.8940961 GlobalBidder_hat{W}__i8,k2 - GlobalBidder_PSI__i8,k2
        + aux_Omega_b8,r12 + aux_Omega_b8,r13 + aux_Omega_b8,r10
        + aux_Omega_b8,r11 + aux_Omega_b8,r5 + aux_Omega_b8,r6
        + aux_Omega_b8,r7 + aux_Omega_b8,r8 + aux_Omega_b8,r9 + aux_Omega_b8,r0
        + aux_Omega_b8,r1 + aux_Omega_b8,r2 + aux_Omega_b8,r3 + aux_Omega_b8,r4
        <= 18497526.8940961
 c1996: 1.1250244065 aux_cap_helper__b1,r6,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b1,r6,band_UNPAIRED__condX1
        - aux_cap_b1,r6,band_UNPAIRED
        - 0.3214355447 aux_cap_helper__b1,r6,band_UNPAIRED__Z2  = 0
 c1997: 5.76668212901648 sv_function_helpervar_b1,r1__Z3
        - sv_function_helpervar_b1,r1__condX3 <= 0
 c1998: - aux_cap_helper__b4,r8,band_LOW_PAIRED__condX2
        + aux_cap_helper__b4,r8,band_LOW_PAIRED__Z2 <= 0
 c1999: - aux_cap_helper__b3,r0,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b3,r0,band_HIGH_PAIRED__condX1 <= 0
 c2000: - aux_Omega_b1,r2 + 138.047432858373 aux_sv_b1,r2  = 0
 c2001: aux_cap_helper__b5,r12,band_LOW_PAIRED__condX2
        + aux_cap_helper__b5,r12,band_LOW_PAIRED__condX1
        - X__b5,r12,band_LOW_PAIRED  = 0
 c2002: - aux_cap_helper__b5,r6,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b5,r6,band_HIGH_PAIRED__condX1 <= 0
 c2003: - sv_function_helpervar_b1,r12__condX3
        + 5.6540979847395 sv_function_helpervar_b1,r12__Z3 <= 0
 c2004: - aux_cap_helper__b5,r1,band_LOW_PAIRED__condX1 <= 0
 c2005: - 0.15994618743616 sv_function_helpervar_b9,r9__Z1
        + sv_function_helpervar_b9,r9__condX1 <= 0
 c2006: - aux_cap_helper__b7,r5,band_LOW_PAIRED__condX1 <= 0
 c2007: - 2 aux_cap_helper__b8,r11,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b8,r11,band_HIGH_PAIRED__condX2 <= 0
 c2008: - aux_sv_b0,r6 + 37.1283877659184 sv_function_helpervar_b0,r6__Z3
        + 16.7450482077216 sv_function_helpervar_b0,r6__Z2
        + 1.747978844 sv_function_helpervar_b0,r6__condX3
        + 5.1789004818 sv_function_helpervar_b0,r6__condX2
        + 109.8706625408 sv_function_helpervar_b0,r6__condX1  = 0
 c2009: - X__b1,r6,band_LOW_PAIRED
        + aux_cap_helper__b1,r6,band_LOW_PAIRED__condX1
        + aux_cap_helper__b1,r6,band_LOW_PAIRED__condX2  = 0
 c2010: - 3 aux_cap_helper__b8,r9,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b8,r9,band_HIGH_PAIRED__condX3 <= 0
 c2011: - sv_function_helpervar_b6,r13__condX3
        + 5.52627638616526 sv_function_helpervar_b6,r13__Z3 <= 0
 c2012: - aux_cap_helper__b0,r0,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b0,r0,band_HIGH_PAIRED__condX1 <= 0
 c2013: 160.764788234054 sv_function_helpervar_b6,r4__Z2
        + 370.053979841674 sv_function_helpervar_b6,r4__Z3
        + 1059.4718675593 sv_function_helpervar_b6,r4__condX1
        + 54.3538912533 sv_function_helpervar_b6,r4__condX2
        + 16.10354371 sv_function_helpervar_b6,r4__condX3 - aux_sv_b6,r4  = 0
 c2014: sv_function_helpervar_b9,r5__condX1
        - 0.159946187454291 sv_function_helpervar_b9,r5__Z1 <= 0
 c2015: aux_cap_helper__b2,r6,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b2,r6,band_HIGH_PAIRED__Z1 <= 0
 c2016: - aux_cap_helper__b2,r13,band_UNPAIRED__condX2
        + aux_cap_helper__b2,r13,band_UNPAIRED__Z2 <= 0
 c2017: aux_cap_helper__b8,r0,band_UNPAIRED__Z1
        + aux_cap_helper__b8,r0,band_UNPAIRED__Z2  = 1
 c2018: - 3 aux_cap_helper__b5,r9,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b5,r9,band_HIGH_PAIRED__condX3 <= 0
 c2019: aux_cap_helper__b5,r3,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b5,r3,band_HIGH_PAIRED__condX2 <= 0
 c2020: sv_function_helpervar_b0,r3__condX2
        - 6.67198907969296 sv_function_helpervar_b0,r3__Z2 <= 0
 c2021: - sv_function_helpervar_b0,r12__condX1 <= 0
 c2022: aux_cap_helper__b2,r2,band_LOW_PAIRED__Z2
        + aux_cap_helper__b2,r2,band_LOW_PAIRED__Z1  = 1
 c2023: aux_cap_helper__b6,r10,band_LOW_PAIRED__Z2
        - aux_cap_helper__b6,r10,band_LOW_PAIRED__condX2 <= 0
 c2024: - aux_cap_helper__b7,r8,band_UNPAIRED__condX2
        + aux_cap_helper__b7,r8,band_UNPAIRED__Z2 <= 0
 c2025: aux_cap_helper__b6,r4,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b6,r4,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b6,r4,band_HIGH_PAIRED__condX1
        - X__b6,r4,band_HIGH_PAIRED  = 0
 c2026: - sv_function_helpervar_b0,r7__condX1 <= 0
 c2027: aux_cap_helper__b4,r8,band_UNPAIRED__condX1
        + aux_cap_helper__b4,r8,band_UNPAIRED__condX2 - X__b4,r8,band_UNPAIRED
         = 0
 c2028: - aux_cap_helper__b5,r3,band_HIGH_PAIRED__condX1 <= 0
 c2029: aux_cap_helper__b5,r5,band_UNPAIRED__Z1
        + aux_cap_helper__b5,r5,band_UNPAIRED__Z2  = 1
 c2030: - aux_cap_helper__b6,r0,band_LOW_PAIRED__condX1 <= 0
 c2031: - aux_cap_b6,r8,band_LOW_PAIRED
        + 3.735962531 aux_cap_helper__b6,r8,band_LOW_PAIRED__condX1
        + 5.2303475434 aux_cap_helper__b6,r8,band_LOW_PAIRED__condX2
        - 1.4943850124 aux_cap_helper__b6,r8,band_LOW_PAIRED__Z2  = 0
 c2032: aux_cap_helper__b8,r9,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b8,r9,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b8,r9,band_HIGH_PAIRED__Z2  = 1
 c2033: aux_cap_helper__b1,r11,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b1,r11,band_HIGH_PAIRED__Z3 <= 0
 c2034: - sv_function_helpervar_b6,r3__condX3
        + 6.06587429996656 sv_function_helpervar_b6,r3__Z3 <= 0
 c2035: aux_cap_helper__b9,r10,band_UNPAIRED__condX1
        - aux_cap_helper__b9,r10,band_UNPAIRED__Z1 <= 0
 c2036: aux_cap_helper__b5,r1,band_LOW_PAIRED__Z2
        - aux_cap_helper__b5,r1,band_LOW_PAIRED__condX2 <= 0
 c2037: aux_cap_helper__b5,r6,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b5,r6,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b5,r6,band_HIGH_PAIRED__Z3  = 1
 c2038: - aux_cap_helper__b0,r10,band_UNPAIRED__condX2
        + aux_cap_helper__b0,r10,band_UNPAIRED__Z2 <= 0
 c2039: sv_function_helpervar_b0,r6__Z3 + sv_function_helpervar_b0,r6__Z1
        + sv_function_helpervar_b0,r6__Z2  = 1
 c2040: - sv_function_helpervar_b2,r8__condX1 <= 0
 c2041: - 2 aux_cap_helper__b3,r2,band_LOW_PAIRED__Z2
        + aux_cap_helper__b3,r2,band_LOW_PAIRED__condX2 <= 0
 c2042: - X__b3,r10,band_UNPAIRED
        + aux_cap_helper__b3,r10,band_UNPAIRED__condX2
        + aux_cap_helper__b3,r10,band_UNPAIRED__condX1  = 0
 c2043: - aux_cap_helper__b4,r8,band_LOW_PAIRED__condX1 <= 0
 c2044: aux_cap_helper__b9,r11,band_LOW_PAIRED__condX1
        - aux_cap_helper__b9,r11,band_LOW_PAIRED__Z1 <= 0
 c2045: - aux_cap_helper__b4,r8,band_UNPAIRED__condX2
        + aux_cap_helper__b4,r8,band_UNPAIRED__Z2 <= 0
 c2046: - sv_function_helpervar_b0,r3__condX2
        + 0.159946187448897 sv_function_helpervar_b0,r3__Z2 <= 0
 c2047: - aux_cap_helper__b0,r10,band_UNPAIRED__condX1 <= 0
 c2048: - 1.4943850124 aux_cap_helper__b3,r6,band_LOW_PAIRED__Z2
        - aux_cap_b3,r6,band_LOW_PAIRED
        + 3.735962531 aux_cap_helper__b3,r6,band_LOW_PAIRED__condX1
        + 5.2303475434 aux_cap_helper__b3,r6,band_LOW_PAIRED__condX2  = 0
 c2049: - aux_cap_helper__b3,r10,band_LOW_PAIRED__condX1 <= 0
 c2050: aux_cap_helper__b9,r12,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b9,r12,band_HIGH_PAIRED__Z2 <= 0
 c2051: aux_cap_helper__b6,r9,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b6,r9,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b6,r9,band_HIGH_PAIRED__Z3  = 1
 c2052: - 0.3214355447 aux_cap_helper__b9,r4,band_UNPAIRED__Z2
        - aux_cap_b9,r4,band_UNPAIRED
        + 0.8035888618 aux_cap_helper__b9,r4,band_UNPAIRED__condX1
        + 1.1250244065 aux_cap_helper__b9,r4,band_UNPAIRED__condX2  = 0
 c2053: 263.660352467213 sv_function_helpervar_b9,r2__Z2
        + 540.617579715035 sv_function_helpervar_b9,r2__Z3
        + 70.6629707388 sv_function_helpervar_b9,r2__condX2
        + 29.870203311 sv_function_helpervar_b9,r2__condX3
        + 1719.0945876917 sv_function_helpervar_b9,r2__condX1 - aux_sv_b9,r2
         = 0
 c2054: sv_function_helpervar_b1,r11__condX2
        - 5.94510944286213 sv_function_helpervar_b1,r11__Z2 <= 0
 c2055: aux_cap_helper__b3,r3,band_LOW_PAIRED__Z2
        - aux_cap_helper__b3,r3,band_LOW_PAIRED__condX2 <= 0
 c2056: aux_cap_helper__b7,r6,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b7,r6,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b7,r6,band_HIGH_PAIRED__Z3  = 1
 c2057: - aux_cap_helper__b6,r11,band_LOW_PAIRED__Z1
        + aux_cap_helper__b6,r11,band_LOW_PAIRED__condX1 <= 0
 c2058: - aux_cap_helper__b6,r11,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b6,r11,band_HIGH_PAIRED__Z2 <= 0
 c2059: sv_function_helpervar_b6,r7__Z3 + sv_function_helpervar_b6,r7__Z2
        + sv_function_helpervar_b6,r7__Z1  = 1
 c2060: - aux_cap_b8,r1,band_LOW_PAIRED
        + 5.2303475434 aux_cap_helper__b8,r1,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b8,r1,band_LOW_PAIRED__condX1
        - 1.4943850124 aux_cap_helper__b8,r1,band_LOW_PAIRED__Z2  = 0
 c2061: - sv_function_helpervar_b9,r11__condX2
        + 0.159946187514989 sv_function_helpervar_b9,r11__Z2 <= 0
 c2062: - sv_function_helpervar_b7,r11__condX1 <= 0
 c2063: sv_function_helpervar_b6,r11__condX2
        - 6.30329523784073 sv_function_helpervar_b6,r11__Z2 <= 0
 c2064: - 2 aux_cap_helper__b9,r0,band_LOW_PAIRED__Z2
        + aux_cap_helper__b9,r0,band_LOW_PAIRED__condX2 <= 0
 c2065: - aux_Omega_b1,r0 + 126.41786711373 aux_sv_b1,r0  = 0
 c2066: aux_cap_b2,r11,band_UNPAIRED + aux_cap_b2,r11,band_HIGH_PAIRED
        - aux_c_b2,r11 + aux_cap_b2,r11,band_LOW_PAIRED  = 0
 c2067: - aux_cap_helper__b8,r0,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b8,r0,band_HIGH_PAIRED__Z2 <= 0
 c2068: aux_cap_b0,r9,band_HIGH_PAIRED + aux_cap_b0,r9,band_UNPAIRED
        + aux_cap_b0,r9,band_LOW_PAIRED - aux_c_b0,r9  = 0
 c2069: aux_cap_b6,r12,band_HIGH_PAIRED + aux_cap_b6,r12,band_LOW_PAIRED
        + aux_cap_b6,r12,band_UNPAIRED - aux_c_b6,r12  = 0
 c2070: aux_cap_helper__b9,r1,band_UNPAIRED__Z1
        + aux_cap_helper__b9,r1,band_UNPAIRED__Z2  = 1
 c2071: sv_function_helpervar_b0,r5__condX1
        + sv_function_helpervar_b0,r5__condX3
        + sv_function_helpervar_b0,r5__condX2 - aux_c_b0,r5  = 0
 c2072: - aux_cap_helper__b6,r8,band_HIGH_PAIRED__condX1 <= 0
 c2073: - aux_Omega_b4,r6 + 35.9854094027298 aux_sv_b4,r6  = 0
 c2074: sv_function_helpervar_b0,r10__condX1
        + sv_function_helpervar_b0,r10__condX3
        + sv_function_helpervar_b0,r10__condX2 - aux_c_b0,r10  = 0
 c2075: - sv_function_helpervar_b2,r0__condX1 <= 0
 c2076: sv_function_helpervar_b4,r12__Z1 + sv_function_helpervar_b4,r12__Z2
        + sv_function_helpervar_b4,r12__Z3  = 1
 c2077: - 2 aux_cap_helper__b6,r5,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b6,r5,band_HIGH_PAIRED__condX2 <= 0
 c2078: aux_cap_helper__b8,r12,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b8,r12,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b8,r12,band_HIGH_PAIRED__Z3  = 1
 c2079: aux_cap_helper__b9,r7,band_UNPAIRED__Z2
        - aux_cap_helper__b9,r7,band_UNPAIRED__condX2 <= 0
 c2080: - aux_cap_helper__b7,r8,band_UNPAIRED__condX1 <= 0
 c2081: - aux_sv_b0,r5 + 109.8706626068 sv_function_helpervar_b0,r5__condX1
        + 1.797330705 sv_function_helpervar_b0,r5__condX3
        + 4.9428731318 sv_function_helpervar_b0,r5__condX2
        + 16.7827998829854 sv_function_helpervar_b0,r5__Z2
        + 36.3390235645993 sv_function_helpervar_b0,r5__Z3  = 0
 c2082: aux_cap_b1,r10,band_UNPAIRED + aux_cap_b1,r10,band_HIGH_PAIRED
        + aux_cap_b1,r10,band_LOW_PAIRED - aux_c_b1,r10  = 0
 c2083: aux_cap_b2,r5,band_HIGH_PAIRED + aux_cap_b2,r5,band_UNPAIRED
        + aux_cap_b2,r5,band_LOW_PAIRED - aux_c_b2,r5  = 0
 c2084: aux_cap_helper__b6,r6,band_UNPAIRED__Z1
        + aux_cap_helper__b6,r6,band_UNPAIRED__Z2  = 1
 c2085: aux_cap_helper__b8,r12,band_LOW_PAIRED__condX1
        - aux_cap_helper__b8,r12,band_LOW_PAIRED__Z1 <= 0
 c2086: - aux_cap_helper__b3,r10,band_LOW_PAIRED__condX2
        + aux_cap_helper__b3,r10,band_LOW_PAIRED__Z2 <= 0
 c2087: - 15.9946187432626 sv_function_helpervar_b5,r6__Z3
        + sv_function_helpervar_b5,r6__condX3 <= 0
 c2088: aux_cap_helper__b1,r12,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b1,r12,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b1,r12,band_HIGH_PAIRED__Z2  = 1
 c2089: - sv_function_helpervar_b0,r6__condX1 <= 0
 c2090: - X__b7,r0,band_LOW_PAIRED
        + aux_cap_helper__b7,r0,band_LOW_PAIRED__condX1
        + aux_cap_helper__b7,r0,band_LOW_PAIRED__condX2  = 0
 c2091: aux_cap_helper__b0,r8,band_UNPAIRED__condX2
        + aux_cap_helper__b0,r8,band_UNPAIRED__condX1 - X__b0,r8,band_UNPAIRED
         = 0
 c2092: 0.0102040816326531 X__b8,r0,band_LOW_PAIRED
        + 0.0102040816326531 X__b8,r0,band_HIGH_PAIRED
        + 0.0102040816326531 X__b8,r0,band_UNPAIRED - GlobalBidder_W_ir__b8,r0
        <= 0
 c2093: - 3 aux_cap_helper__b1,r1,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b1,r1,band_HIGH_PAIRED__condX3 <= 0
 c2094: - aux_cap_helper__b8,r3,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b8,r3,band_HIGH_PAIRED__Z2 <= 0
 c2095: X__b7,r11,band_LOW_PAIRED + X__b0,r11,band_LOW_PAIRED
        + X__b3,r11,band_LOW_PAIRED + X__b6,r11,band_LOW_PAIRED
        + X__b4,r11,band_LOW_PAIRED + X__b8,r11,band_LOW_PAIRED
        + X__b5,r11,band_LOW_PAIRED + X__b2,r11,band_LOW_PAIRED
        + X__b1,r11,band_LOW_PAIRED + X__b9,r11,band_LOW_PAIRED <= 2
 c2096: aux_cap_helper__b5,r12,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b5,r12,band_HIGH_PAIRED__Z3 <= 0
 c2097: - 1.4943850124 aux_cap_helper__b4,r0,band_LOW_PAIRED__Z2
        - aux_cap_b4,r0,band_LOW_PAIRED
        + 3.735962531 aux_cap_helper__b4,r0,band_LOW_PAIRED__condX1
        + 5.2303475434 aux_cap_helper__b4,r0,band_LOW_PAIRED__condX2  = 0
 c2098: - 6.33877630346304 sv_function_helpervar_b7,r8__Z2
        + sv_function_helpervar_b7,r8__condX2 <= 0
 c2099: - aux_cap_helper__b8,r3,band_HIGH_PAIRED__condX1 <= 0
 c2100: - X__b4,r10,band_LOW_PAIRED
        + aux_cap_helper__b4,r10,band_LOW_PAIRED__condX2
        + aux_cap_helper__b4,r10,band_LOW_PAIRED__condX1  = 0
 c2101: sv_function_helpervar_b6,r3__Z1 + sv_function_helpervar_b6,r3__Z3
        + sv_function_helpervar_b6,r3__Z2  = 1
 c2102: 2 aux_cap_helper__b1,r4,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b1,r4,band_HIGH_PAIRED__condX3 <= 0
 c2103: sv_function_helpervar_b2,r8__condX3
        - 15.9946187432626 sv_function_helpervar_b2,r8__Z3 <= 0
 c2104: aux_cap_helper__b6,r0,band_LOW_PAIRED__Z1
        + aux_cap_helper__b6,r0,band_LOW_PAIRED__Z2  = 1
 c2105: sv_function_helpervar_b7,r7__condX2
        - 7.00008283627391 sv_function_helpervar_b7,r7__Z2 <= 0
 c2106: aux_cap_helper__b5,r3,band_LOW_PAIRED__condX2
        + aux_cap_helper__b5,r3,band_LOW_PAIRED__condX1
        - X__b5,r3,band_LOW_PAIRED  = 0
 c2107: - aux_cap_helper__b2,r13,band_UNPAIRED__condX1 <= 0
 c2108: - 6.32530579601375 sv_function_helpervar_b4,r3__Z2
        + sv_function_helpervar_b4,r3__condX2 <= 0
 c2109: - X__b1,r12,band_HIGH_PAIRED
        + aux_cap_helper__b1,r12,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b1,r12,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b1,r12,band_HIGH_PAIRED__condX1  = 0
 c2110: aux_cap_helper__b7,r9,band_LOW_PAIRED__Z2
        + aux_cap_helper__b7,r9,band_LOW_PAIRED__Z1  = 1
 c2111: aux_cap_helper__b1,r7,band_LOW_PAIRED__condX1
        + aux_cap_helper__b1,r7,band_LOW_PAIRED__condX2
        - X__b1,r7,band_LOW_PAIRED  = 0
 c2112: X__b8,r5,band_LOW_PAIRED + X__b6,r5,band_LOW_PAIRED
        + X__b4,r5,band_LOW_PAIRED + X__b9,r5,band_LOW_PAIRED
        + X__b7,r5,band_LOW_PAIRED + X__b5,r5,band_LOW_PAIRED
        + X__b2,r5,band_LOW_PAIRED + X__b3,r5,band_LOW_PAIRED
        + X__b0,r5,band_LOW_PAIRED + X__b1,r5,band_LOW_PAIRED <= 2
 c2113: - aux_cap_helper__b0,r13,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b0,r13,band_HIGH_PAIRED__Z3 <= 0
 c2114: - sv_function_helpervar_b0,r5__condX3
        + 6.21712284501458 sv_function_helpervar_b0,r5__Z3 <= 0
 c2115: sv_function_helpervar_b0,r12__condX3
        - 15.9946187432626 sv_function_helpervar_b0,r12__Z3 <= 0
 c2116: aux_cap_helper__b2,r9,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b2,r9,band_HIGH_PAIRED__Z1 <= 0
 c2117: aux_cap_helper__b8,r2,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b8,r2,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b8,r2,band_HIGH_PAIRED__condX1
        - X__b8,r2,band_HIGH_PAIRED  = 0
 c2118: - X__b2,r9,band_HIGH_PAIRED
        + aux_cap_helper__b2,r9,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b2,r9,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b2,r9,band_HIGH_PAIRED__condX1  = 0
 c2119: aux_cap_helper__b5,r12,band_LOW_PAIRED__condX1
        - aux_cap_helper__b5,r12,band_LOW_PAIRED__Z1 <= 0
 c2120: aux_cap_b6,r9,band_HIGH_PAIRED - aux_c_b6,r9
        + aux_cap_b6,r9,band_UNPAIRED + aux_cap_b6,r9,band_LOW_PAIRED  = 0
 c2121: aux_cap_helper__b8,r7,band_LOW_PAIRED__Z2
        - aux_cap_helper__b8,r7,band_LOW_PAIRED__condX2 <= 0
 c2122: - aux_cap_helper__b3,r8,band_HIGH_PAIRED__condX1 <= 0
 c2123: - aux_Omega_b3,r9 + 223.241592871955 aux_sv_b3,r9  = 0
 c2124: - aux_cap_helper__b7,r5,band_LOW_PAIRED__condX2
        + aux_cap_helper__b7,r5,band_LOW_PAIRED__Z2 <= 0
 c2125: aux_cap_helper__b0,r6,band_LOW_PAIRED__condX1
        - aux_cap_helper__b0,r6,band_LOW_PAIRED__Z1 <= 0
 c2126: aux_cap_helper__b8,r13,band_LOW_PAIRED__Z2
        + aux_cap_helper__b8,r13,band_LOW_PAIRED__Z1  = 1
 c2127: - X__b7,r10,band_HIGH_PAIRED
        + aux_cap_helper__b7,r10,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b7,r10,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b7,r10,band_HIGH_PAIRED__condX2  = 0
 c2128: - aux_cap_helper__b0,r10,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b0,r10,band_HIGH_PAIRED__Z3 <= 0
 c2129: X__b7,r3,band_LOW_PAIRED - GlobalBidder_W_ir__b7,r3
        + X__b7,r3,band_UNPAIRED + X__b7,r3,band_HIGH_PAIRED >= 0
 c2130: - aux_c_b7,r5 + sv_function_helpervar_b7,r5__condX3
        + sv_function_helpervar_b7,r5__condX2
        + sv_function_helpervar_b7,r5__condX1  = 0
 c2131: - aux_cap_helper__b9,r7,band_UNPAIRED__condX1 <= 0
 c2132: - aux_cap_helper__b8,r7,band_LOW_PAIRED__condX1 <= 0
 c2133: - sv_function_helpervar_b8,r6__condX2
        + 0.159946187438625 sv_function_helpervar_b8,r6__Z2 <= 0
 c2134: 6.59858228882779 sv_function_helpervar_b0,r11__Z3
        - sv_function_helpervar_b0,r11__condX3 <= 0
 c2135: aux_cap_helper__b3,r10,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b3,r10,band_UNPAIRED__Z2 <= 0
 c2136: 2 aux_cap_helper__b1,r7,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b1,r7,band_HIGH_PAIRED__condX3 <= 0
 c2137: - 3 aux_cap_helper__b7,r4,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b7,r4,band_HIGH_PAIRED__condX3 <= 0
 c2138: 2 aux_cap_helper__b4,r4,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b4,r4,band_HIGH_PAIRED__condX3 <= 0
 c2139: - 2 aux_cap_helper__b9,r5,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b9,r5,band_HIGH_PAIRED__condX2 <= 0
 c2140: aux_cap_helper__b3,r8,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b3,r8,band_HIGH_PAIRED__condX2 <= 0
 c2141: sv_function_helpervar_b5,r5__condX3
        - 15.9946187432626 sv_function_helpervar_b5,r5__Z3 <= 0
 c2142: aux_cap_helper__b6,r8,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b6,r8,band_HIGH_PAIRED__condX2 <= 0
 c2143: aux_cap_helper__b5,r2,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b5,r2,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b5,r2,band_HIGH_PAIRED__condX2
        - X__b5,r2,band_HIGH_PAIRED  = 0
 c2144: - sv_function_helpervar_b2,r3__condX3
        + 6.24106605787985 sv_function_helpervar_b2,r3__Z3 <= 0
 c2145: aux_cap_helper__b7,r3,band_UNPAIRED__Z2
        - aux_cap_helper__b7,r3,band_UNPAIRED__condX2 <= 0
 c2146: sv_function_helpervar_b1,r13__condX3
        + sv_function_helpervar_b1,r13__condX2
        + sv_function_helpervar_b1,r13__condX1 - aux_c_b1,r13  = 0
 c2147: aux_cap_helper__b2,r12,band_LOW_PAIRED__Z1
        + aux_cap_helper__b2,r12,band_LOW_PAIRED__Z2  = 1
 c2148: - 3 aux_cap_helper__b7,r1,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b7,r1,band_HIGH_PAIRED__condX3 <= 0
 c2149: - aux_cap_helper__b7,r8,band_LOW_PAIRED__condX1 <= 0
 c2150: - aux_cap_helper__b2,r5,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b2,r5,band_HIGH_PAIRED__Z3 <= 0
 c2151: - 0.3214355447 aux_cap_helper__b9,r7,band_UNPAIRED__Z2
        + 1.1250244065 aux_cap_helper__b9,r7,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b9,r7,band_UNPAIRED__condX1
        - aux_cap_b9,r7,band_UNPAIRED  = 0
 c2152: - aux_cap_helper__b6,r13,band_LOW_PAIRED__condX1 <= 0
 c2153: - aux_cap_helper__b2,r1,band_LOW_PAIRED__condX1 <= 0
 c2154: - aux_cap_helper__b7,r3,band_UNPAIRED__condX1 <= 0
 c2155: X__b9,r12,band_LOW_PAIRED - GlobalBidder_W_ir__b9,r12
        + X__b9,r12,band_UNPAIRED + X__b9,r12,band_HIGH_PAIRED >= 0
 c2156: - aux_cap_helper__b3,r0,band_LOW_PAIRED__condX1 <= 0
 c2157: 0.159946187506737 sv_function_helpervar_b0,r7__Z2
        - sv_function_helpervar_b0,r7__condX2 <= 0
 c2158: aux_cap_b1,r2,band_HIGH_PAIRED + aux_cap_b1,r2,band_UNPAIRED
        - aux_c_b1,r2 + aux_cap_b1,r2,band_LOW_PAIRED  = 0
 c2159: 0.159946187432645 sv_function_helpervar_b8,r5__Z2
        - sv_function_helpervar_b8,r5__condX2 <= 0
 c2160: sv_function_helpervar_b4,r8__condX1
        - 0.159946187453367 sv_function_helpervar_b4,r8__Z1 <= 0
 c2161: - aux_cap_helper__b1,r8,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b1,r8,band_HIGH_PAIRED__condX1 <= 0
 c2162: aux_cap_helper__b0,r11,band_LOW_PAIRED__condX1
        - aux_cap_helper__b0,r11,band_LOW_PAIRED__Z1 <= 0
 c2163: - aux_c_b0,r6 + sv_function_helpervar_b0,r6__condX3
        + sv_function_helpervar_b0,r6__condX2
        + sv_function_helpervar_b0,r6__condX1  = 0
 c2164: 1.1250244065 aux_cap_helper__b3,r10,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b3,r10,band_UNPAIRED__condX1
        - aux_cap_b3,r10,band_UNPAIRED
        - 0.3214355447 aux_cap_helper__b3,r10,band_UNPAIRED__Z2  = 0
 c2165: - aux_cap_helper__b4,r1,band_UNPAIRED__condX1 <= 0
 c2166: aux_cap_helper__b8,r8,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b8,r8,band_UNPAIRED__Z2 <= 0
 c2167: - aux_c_b2,r0 + aux_cap_b2,r0,band_UNPAIRED
        + aux_cap_b2,r0,band_LOW_PAIRED + aux_cap_b2,r0,band_HIGH_PAIRED  = 0
 c2168: 456.133911036645 sv_function_helpervar_b3,r5__Z3
        + 210.739786140347 sv_function_helpervar_b3,r5__Z2 - aux_sv_b3,r5
        + 1379.5885617819 sv_function_helpervar_b3,r5__condX1
        + 62.0217630854 sv_function_helpervar_b3,r5__condX2
        + 22.5779118148 sv_function_helpervar_b3,r5__condX3  = 0
 c2169: - X__b0,r4,band_LOW_PAIRED
        + aux_cap_helper__b0,r4,band_LOW_PAIRED__condX1
        + aux_cap_helper__b0,r4,band_LOW_PAIRED__condX2  = 0
 c2170: - aux_cap_helper__b6,r9,band_LOW_PAIRED__condX2
        + aux_cap_helper__b6,r9,band_LOW_PAIRED__Z2 <= 0
 c2171: aux_cap_helper__b1,r3,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b1,r3,band_UNPAIRED__Z2 <= 0
 c2172: - 1.4943850124 aux_cap_helper__b1,r10,band_LOW_PAIRED__Z2
        - aux_cap_b1,r10,band_LOW_PAIRED
        + 5.2303475434 aux_cap_helper__b1,r10,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b1,r10,band_LOW_PAIRED__condX1  = 0
 c2173: - 2 aux_cap_helper__b2,r10,band_UNPAIRED__Z2
        + aux_cap_helper__b2,r10,band_UNPAIRED__condX2 <= 0
 c2174: 283.304802251947 sv_function_helpervar_b7,r4__Z2
        + 595.631809066121 sv_function_helpervar_b7,r4__Z3
        + 1850.3233256734 sv_function_helpervar_b7,r4__condX1
        + 79.0725897162 sv_function_helpervar_b7,r4__condX2
        + 31.2909807774 sv_function_helpervar_b7,r4__condX3 - aux_sv_b7,r4  = 0
 c2175: GlobalBidder_W_i_7 + 14 GlobalBidder_hat{W}__i7,k2 <= 26
 c2176: aux_cap_helper__b5,r6,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b5,r6,band_UNPAIRED__Z2 <= 0
 c2177: 161.29429279429 sv_function_helpervar_b6,r8__Z2
        + 361.336475422617 sv_function_helpervar_b6,r8__Z3
        + 16.6485710444 sv_function_helpervar_b6,r8__condX3
        + 1059.4718673293 sv_function_helpervar_b6,r8__condX1
        + 51.0433743195 sv_function_helpervar_b6,r8__condX2 - aux_sv_b6,r8  = 0
 c2178: - X__b3,r4,band_UNPAIRED + aux_cap_helper__b3,r4,band_UNPAIRED__condX1
        + aux_cap_helper__b3,r4,band_UNPAIRED__condX2  = 0
 c2179: - aux_cap_helper__b6,r9,band_LOW_PAIRED__condX1 <= 0
 c2180: - aux_cap_helper__b3,r1,band_HIGH_PAIRED__condX1 <= 0
 c2181: aux_cap_helper__b5,r13,band_UNPAIRED__condX2
        + aux_cap_helper__b5,r13,band_UNPAIRED__condX1
        - X__b5,r13,band_UNPAIRED  = 0
 c2182: aux_cap_helper__b7,r8,band_LOW_PAIRED__Z2
        - aux_cap_helper__b7,r8,band_LOW_PAIRED__condX2 <= 0
 c2183: aux_cap_b0,r4,band_UNPAIRED - aux_c_b0,r4
        + aux_cap_b0,r4,band_HIGH_PAIRED + aux_cap_b0,r4,band_LOW_PAIRED  = 0
 c2184: 41.8182647586564 sv_function_helpervar_b1,r10__Z3
        + 19.1667763907994 sv_function_helpervar_b1,r10__Z2 - aux_sv_b1,r10
        + 125.5635947379 sv_function_helpervar_b1,r10__condX1
        + 5.7309391537 sv_function_helpervar_b1,r10__condX2
        + 2.0359826235 sv_function_helpervar_b1,r10__condX3  = 0
 c2185: aux_cap_helper__b2,r11,band_LOW_PAIRED__Z2
        - aux_cap_helper__b2,r11,band_LOW_PAIRED__condX2 <= 0
 c2186: aux_cap_helper__b2,r9,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b2,r9,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b2,r9,band_HIGH_PAIRED__Z3  = 1
 c2187: sv_function_helpervar_b8,r11__condX3
        + sv_function_helpervar_b8,r11__condX1
        + sv_function_helpervar_b8,r11__condX2 - aux_c_b8,r11  = 0
 c2188: 16.7491972486 sv_function_helpervar_b6,r1__condX3
        + 50.4974220869 sv_function_helpervar_b6,r1__condX2
        + 1059.4718672864 sv_function_helpervar_b6,r1__condX1
        + 161.3816157721 sv_function_helpervar_b6,r1__Z2
        + 359.726997651699 sv_function_helpervar_b6,r1__Z3 - aux_sv_b6,r1  = 0
 c2189: - 2 aux_cap_helper__b0,r7,band_UNPAIRED__Z2
        + aux_cap_helper__b0,r7,band_UNPAIRED__condX2 <= 0
 c2190: - 2 aux_cap_helper__b8,r4,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b8,r4,band_HIGH_PAIRED__condX2 <= 0
 c2191: - aux_cap_helper__b1,r12,band_LOW_PAIRED__condX1 <= 0
 c2192: aux_cap_helper__b2,r1,band_LOW_PAIRED__Z2
        - aux_cap_helper__b2,r1,band_LOW_PAIRED__condX2 <= 0
 c2193: - sv_function_helpervar_b2,r13__condX1 <= 0
 c2194: - aux_cap_helper__b3,r0,band_LOW_PAIRED__condX2
        + aux_cap_helper__b3,r0,band_LOW_PAIRED__Z2 <= 0
 c2195: - aux_cap_helper__b9,r1,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b9,r1,band_HIGH_PAIRED__Z2 <= 0
 c2196: X__b7,r11,band_UNPAIRED + X__b7,r11,band_LOW_PAIRED
        + X__b7,r11,band_HIGH_PAIRED - GlobalBidder_W_ir__b7,r11 >= 0
 c2197: sv_function_helpervar_b7,r12__condX2
        - 7.70995398967424 sv_function_helpervar_b7,r12__Z2 <= 0
 c2198: aux_cap_b7,r7,band_LOW_PAIRED - aux_c_b7,r7
        + aux_cap_b7,r7,band_UNPAIRED + aux_cap_b7,r7,band_HIGH_PAIRED  = 0
 c2199: sv_function_helpervar_b9,r4__condX1
        + sv_function_helpervar_b9,r4__condX3
        + sv_function_helpervar_b9,r4__condX2 - aux_c_b9,r4  = 0
 c2200: - 2 aux_cap_helper__b5,r11,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b5,r11,band_HIGH_PAIRED__condX2 <= 0
 c2201: aux_cap_helper__b7,r12,band_UNPAIRED__condX1
        - aux_cap_helper__b7,r12,band_UNPAIRED__Z1 <= 0
 c2202: aux_cap_b9,r3,band_HIGH_PAIRED + aux_cap_b9,r3,band_UNPAIRED
        + aux_cap_b9,r3,band_LOW_PAIRED - aux_c_b9,r3  = 0
 c2203: 0.159946187444592 sv_function_helpervar_b4,r10__Z2
        - sv_function_helpervar_b4,r10__condX2 <= 0
 c2204: - aux_cap_helper__b9,r10,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b9,r10,band_HIGH_PAIRED__condX1 <= 0
 c2205: aux_cap_helper__b6,r10,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b6,r10,band_HIGH_PAIRED__Z3 <= 0
 c2206: - 2 aux_cap_helper__b6,r2,band_UNPAIRED__Z2
        + aux_cap_helper__b6,r2,band_UNPAIRED__condX2 <= 0
 c2207: 0.159946187433409 sv_function_helpervar_b3,r5__Z2
        - sv_function_helpervar_b3,r5__condX2 <= 0
 c2208: aux_cap_helper__b5,r13,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b5,r13,band_UNPAIRED__Z2 <= 0
 c2209: - aux_cap_b0,r11,band_HIGH_PAIRED
        + 1.3908260184 aux_cap_helper__b0,r11,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b0,r11,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b0,r11,band_HIGH_PAIRED__condX1
        - 0.6181448971 aux_cap_helper__b0,r11,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b0,r11,band_HIGH_PAIRED__Z3  = 0
 c2210: 5.8731634253912 sv_function_helpervar_b6,r10__Z3
        - sv_function_helpervar_b6,r10__condX3 <= 0
 c2211: 5.6205873774191 sv_function_helpervar_b6,r5__Z3
        - sv_function_helpervar_b6,r5__condX3 <= 0
 c2212: - aux_cap_helper__b9,r4,band_LOW_PAIRED__Z1
        + aux_cap_helper__b9,r4,band_LOW_PAIRED__condX1 <= 0
 c2213: - 6.32727934049225 sv_function_helpervar_b4,r10__Z2
        + sv_function_helpervar_b4,r10__condX2 <= 0
 c2214: - 15.9946187432626 sv_function_helpervar_b2,r12__Z3
        + sv_function_helpervar_b2,r12__condX3 <= 0
 c2215: - aux_Omega_b7,r2 + 354.843754132788 aux_sv_b7,r2  = 0
 c2216: - sv_function_helpervar_b5,r8__condX3
        + 6.3822247076733 sv_function_helpervar_b5,r8__Z3 <= 0
 c2217: aux_cap_helper__b9,r7,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b9,r7,band_HIGH_PAIRED__Z1 <= 0
 c2218: aux_cap_helper__b7,r8,band_LOW_PAIRED__condX2
        + aux_cap_helper__b7,r8,band_LOW_PAIRED__condX1
        - X__b7,r8,band_LOW_PAIRED  = 0
 c2219: aux_cap_helper__b7,r11,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b7,r11,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b7,r11,band_HIGH_PAIRED__Z3  = 1
 c2220: - 2 aux_cap_helper__b3,r5,band_LOW_PAIRED__Z2
        + aux_cap_helper__b3,r5,band_LOW_PAIRED__condX2 <= 0
 c2221: aux_cap_helper__b6,r2,band_LOW_PAIRED__Z2
        + aux_cap_helper__b6,r2,band_LOW_PAIRED__Z1  = 1
 c2222: aux_cap_helper__b6,r13,band_LOW_PAIRED__Z2
        - aux_cap_helper__b6,r13,band_LOW_PAIRED__condX2 <= 0
 c2223: aux_cap_helper__b8,r9,band_UNPAIRED__Z2
        + aux_cap_helper__b8,r9,band_UNPAIRED__Z1  = 1
 c2224: - aux_c_b5,r13 + sv_function_helpervar_b5,r13__condX1
        + sv_function_helpervar_b5,r13__condX3
        + sv_function_helpervar_b5,r13__condX2  = 0
 c2225: - aux_c_b8,r5 + aux_cap_b8,r5,band_HIGH_PAIRED
        + aux_cap_b8,r5,band_UNPAIRED + aux_cap_b8,r5,band_LOW_PAIRED  = 0
 c2226: aux_cap_helper__b9,r6,band_LOW_PAIRED__condX1
        + aux_cap_helper__b9,r6,band_LOW_PAIRED__condX2
        - X__b9,r6,band_LOW_PAIRED  = 0
 c2227: sv_function_helpervar_b4,r9__Z1 + sv_function_helpervar_b4,r9__Z3
        + sv_function_helpervar_b4,r9__Z2  = 1
 c2228: aux_cap_helper__b2,r4,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b2,r4,band_HIGH_PAIRED__Z2 <= 0
 c2229: aux_cap_helper__b0,r10,band_LOW_PAIRED__Z1
        + aux_cap_helper__b0,r10,band_LOW_PAIRED__Z2  = 1
 c2230: - aux_cap_helper__b3,r5,band_UNPAIRED__condX2
        + aux_cap_helper__b3,r5,band_UNPAIRED__Z2 <= 0
 c2231: 544.934241051702 aux_sv_b4,r12 - aux_Omega_b4,r12  = 0
 c2232: - X__b6,r3,band_UNPAIRED + aux_cap_helper__b6,r3,band_UNPAIRED__condX2
        + aux_cap_helper__b6,r3,band_UNPAIRED__condX1  = 0
 c2233: - aux_cap_b1,r10,band_UNPAIRED
        - 0.3214355447 aux_cap_helper__b1,r10,band_UNPAIRED__Z2
        + 1.1250244065 aux_cap_helper__b1,r10,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b1,r10,band_UNPAIRED__condX1  = 0
 c2234: aux_cap_helper__b4,r7,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b4,r7,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b4,r7,band_HIGH_PAIRED__Z3  = 1
 c2235: - aux_cap_helper__b7,r6,band_LOW_PAIRED__Z1
        + aux_cap_helper__b7,r6,band_LOW_PAIRED__condX1 <= 0
 c2236: - aux_cap_helper__b7,r12,band_LOW_PAIRED__condX1 <= 0
 c2237: - sv_function_helpervar_b4,r1__condX1 <= 0
 c2238: - 3 aux_cap_helper__b5,r2,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b5,r2,band_HIGH_PAIRED__condX3 <= 0
 c2239: - aux_cap_helper__b0,r6,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b0,r6,band_HIGH_PAIRED__Z3 <= 0
 c2240: - 1.4943850124 aux_cap_helper__b2,r13,band_LOW_PAIRED__Z2
        + 5.2303475434 aux_cap_helper__b2,r13,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b2,r13,band_LOW_PAIRED__condX1
        - aux_cap_b2,r13,band_LOW_PAIRED  = 0
 c2241: - X__b8,r5,band_HIGH_PAIRED
        + aux_cap_helper__b8,r5,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b8,r5,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b8,r5,band_HIGH_PAIRED__condX3  = 0
 c2242: - aux_cap_helper__b1,r9,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b1,r9,band_HIGH_PAIRED__Z2 <= 0
 c2243: aux_cap_helper__b1,r11,band_UNPAIRED__condX1
        - aux_cap_helper__b1,r11,band_UNPAIRED__Z1 <= 0
 c2244: - sv_function_helpervar_b1,r9__condX2
        + 0.159946187442558 sv_function_helpervar_b1,r9__Z2 <= 0
 c2245: - 5.45409675243343 sv_function_helpervar_b4,r12__Z2
        + sv_function_helpervar_b4,r12__condX2 <= 0
 c2246: - aux_cap_helper__b1,r10,band_LOW_PAIRED__Z1
        + aux_cap_helper__b1,r10,band_LOW_PAIRED__condX1 <= 0
 c2247: aux_cap_helper__b1,r4,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b1,r4,band_LOW_PAIRED__Z2 <= 0
 c2248: 0.159946187458717 sv_function_helpervar_b3,r0__Z2
        - sv_function_helpervar_b3,r0__condX2 <= 0
 c2249: aux_cap_helper__b9,r5,band_UNPAIRED__condX1
        - aux_cap_helper__b9,r5,band_UNPAIRED__Z1 <= 0
 c2250: - X__b2,r2,band_LOW_PAIRED
        + aux_cap_helper__b2,r2,band_LOW_PAIRED__condX1
        + aux_cap_helper__b2,r2,band_LOW_PAIRED__condX2  = 0
 c2251: - aux_cap_helper__b8,r11,band_LOW_PAIRED__condX2
        + aux_cap_helper__b8,r11,band_LOW_PAIRED__Z2 <= 0
 c2252: - 5.60347911969823 sv_function_helpervar_b4,r1__Z2
        + sv_function_helpervar_b4,r1__condX2 <= 0
 c2253: sv_function_helpervar_b4,r13__condX3
        - 15.9946187432626 sv_function_helpervar_b4,r13__Z3 <= 0
 c2254: 56.5299470701 sv_function_helpervar_b4,r5__condX2
        + 21.4379939459 sv_function_helpervar_b4,r5__condX3
        + 1289.327263572 sv_function_helpervar_b4,r5__condX1 - aux_sv_b4,r5
        + 197.181230651693 sv_function_helpervar_b4,r5__Z2
        + 420.896275627834 sv_function_helpervar_b4,r5__Z3  = 0
 c2255: aux_cap_helper__b1,r12,band_LOW_PAIRED__Z2
        - aux_cap_helper__b1,r12,band_LOW_PAIRED__condX2 <= 0
 c2256: - X__b2,r9,band_LOW_PAIRED
        + aux_cap_helper__b2,r9,band_LOW_PAIRED__condX1
        + aux_cap_helper__b2,r9,band_LOW_PAIRED__condX2  = 0
 c2257: - sv_function_helpervar_b3,r8__condX2
        + 0.159946187466617 sv_function_helpervar_b3,r8__Z2 <= 0
 c2258: 1.3908260184 aux_cap_helper__b4,r9,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b4,r9,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b4,r9,band_HIGH_PAIRED__condX1
        - 0.6181448971 aux_cap_helper__b4,r9,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b4,r9,band_HIGH_PAIRED__Z3
        - aux_cap_b4,r9,band_HIGH_PAIRED  = 0
 c2259: aux_cap_helper__b7,r8,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b7,r8,band_HIGH_PAIRED__Z1 <= 0
 c2260: - 15.9946187432626 sv_function_helpervar_b0,r6__Z3
        + sv_function_helpervar_b0,r6__condX3 <= 0
 c2261: - aux_cap_b5,r10,band_UNPAIRED
        - 0.3214355447 aux_cap_helper__b5,r10,band_UNPAIRED__Z2
        + 1.1250244065 aux_cap_helper__b5,r10,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b5,r10,band_UNPAIRED__condX1  = 0
 c2262: aux_cap_helper__b9,r8,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b9,r8,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b9,r8,band_HIGH_PAIRED__Z1  = 1
 c2263: - 6.78937087899278 sv_function_helpervar_b9,r2__Z2
        + sv_function_helpervar_b9,r2__condX2 <= 0
 c2264: 2 aux_cap_helper__b8,r5,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b8,r5,band_HIGH_PAIRED__condX3 <= 0
 c2265: - sv_function_helpervar_b7,r13__condX2
        + 0.159946187553672 sv_function_helpervar_b7,r13__Z2 <= 0
 c2266: - aux_cap_helper__b8,r2,band_LOW_PAIRED__Z1
        + aux_cap_helper__b8,r2,band_LOW_PAIRED__condX1 <= 0
 c2267: - aux_c_b4,r11 + sv_function_helpervar_b4,r11__condX2
        + sv_function_helpervar_b4,r11__condX1
        + sv_function_helpervar_b4,r11__condX3  = 0
 c2268: 178.254083179305 sv_function_helpervar_b5,r3__Z2
        + 395.252476812299 sv_function_helpervar_b5,r3__Z3 - aux_sv_b5,r3
        + 18.6068327798 sv_function_helpervar_b5,r3__condX3
        + 1169.5974418095 sv_function_helpervar_b5,r3__condX1
        + 55.1345968694 sv_function_helpervar_b5,r3__condX2  = 0
 c2269: - aux_cap_helper__b1,r2,band_HIGH_PAIRED__condX1 <= 0
 c2270: 0.159946187475059 sv_function_helpervar_b5,r13__Z2
        - sv_function_helpervar_b5,r13__condX2 <= 0
 c2271: - aux_cap_helper__b5,r10,band_HIGH_PAIRED__condX1 <= 0
 c2272: - aux_cap_helper__b7,r12,band_LOW_PAIRED__condX2
        + aux_cap_helper__b7,r12,band_LOW_PAIRED__Z2 <= 0
 c2273: - aux_cap_helper__b6,r12,band_UNPAIRED__Z1
        + aux_cap_helper__b6,r12,band_UNPAIRED__condX1 <= 0
 c2274: - aux_cap_helper__b9,r0,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b9,r0,band_HIGH_PAIRED__condX1 <= 0
 c2275: aux_cap_helper__b0,r5,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b0,r5,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b0,r5,band_HIGH_PAIRED__condX2
        - X__b0,r5,band_HIGH_PAIRED  = 0
 c2276: - aux_cap_b4,r8,band_LOW_PAIRED
        + 5.2303475434 aux_cap_helper__b4,r8,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b4,r8,band_LOW_PAIRED__condX1
        - 1.4943850124 aux_cap_helper__b4,r8,band_LOW_PAIRED__Z2  = 0
 c2277: - aux_cap_helper__b2,r11,band_LOW_PAIRED__condX1 <= 0
 c2278: - 2 aux_cap_helper__b9,r4,band_UNPAIRED__Z2
        + aux_cap_helper__b9,r4,band_UNPAIRED__condX2 <= 0
 c2279: aux_cap_helper__b7,r6,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b7,r6,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b7,r6,band_HIGH_PAIRED__condX1
        - X__b7,r6,band_HIGH_PAIRED  = 0
 c2280: aux_cap_helper__b7,r13,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b7,r13,band_HIGH_PAIRED__Z2 <= 0
 c2281: - aux_cap_helper__b4,r1,band_UNPAIRED__condX2
        + aux_cap_helper__b4,r1,band_UNPAIRED__Z2 <= 0
 c2282: - aux_cap_helper__b3,r13,band_UNPAIRED__condX1 <= 0
 c2283: aux_cap_helper__b0,r4,band_LOW_PAIRED__Z2
        + aux_cap_helper__b0,r4,band_LOW_PAIRED__Z1  = 1
 c2284: aux_cap_helper__b2,r9,band_UNPAIRED__Z2
        - aux_cap_helper__b2,r9,band_UNPAIRED__condX2 <= 0
 c2285: - aux_Omega_b3,r13 + 17.9998205078567 aux_sv_b3,r13  = 0
 c2286: - 15.9946187432626 sv_function_helpervar_b4,r0__Z3
        + sv_function_helpervar_b4,r0__condX3 <= 0
 c2287: sv_function_helpervar_b8,r4__condX1
        - 0.159946187488937 sv_function_helpervar_b8,r4__Z1 <= 0
 c2288: sv_function_helpervar_b3,r3__condX1
        - 0.159946187517238 sv_function_helpervar_b3,r3__Z1 <= 0
 c2289: - 0.15994618747751 sv_function_helpervar_b3,r2__Z1
        + sv_function_helpervar_b3,r2__condX1 <= 0
 c2290: aux_cap_b3,r11,band_UNPAIRED - aux_c_b3,r11
        + aux_cap_b3,r11,band_HIGH_PAIRED + aux_cap_b3,r11,band_LOW_PAIRED  = 0
 c2291: aux_cap_helper__b2,r6,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b2,r6,band_LOW_PAIRED__Z2 <= 0
 c2292: - aux_cap_helper__b2,r9,band_UNPAIRED__condX1 <= 0
 c2293: aux_cap_helper__b7,r11,band_LOW_PAIRED__Z2
        + aux_cap_helper__b7,r11,band_LOW_PAIRED__Z1  = 1
 c2294: sv_function_helpervar_b8,r8__Z3 + sv_function_helpervar_b8,r8__Z2
        + sv_function_helpervar_b8,r8__Z1  = 1
 c2295: - aux_cap_helper__b3,r5,band_UNPAIRED__condX1 <= 0
 c2296: aux_cap_helper__b7,r2,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b7,r2,band_HIGH_PAIRED__condX2 <= 0
 c2297: GlobalBidder_W_i_7 + 14 GlobalBidder_hat{W}__i7,k0 <= 28
 c2298: - 2 aux_cap_helper__b2,r11,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b2,r11,band_HIGH_PAIRED__condX2 <= 0
 c2299: - aux_c_b7,r12 + aux_cap_b7,r12,band_UNPAIRED
        + aux_cap_b7,r12,band_HIGH_PAIRED + aux_cap_b7,r12,band_LOW_PAIRED  = 0
 c2300: - aux_sv_b8,r8 + 23.0297131483 sv_function_helpervar_b8,r8__condX3
        + 56.2409741039 sv_function_helpervar_b8,r8__condX2
        + 1343.0085750373 sv_function_helpervar_b8,r8__condX1
        + 427.237782695843 sv_function_helpervar_b8,r8__Z3
        + 205.813571963528 sv_function_helpervar_b8,r8__Z2  = 0
 c2301: - aux_cap_helper__b1,r2,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b1,r2,band_HIGH_PAIRED__Z2 <= 0
 c2302: aux_cap_helper__b0,r5,band_UNPAIRED__condX1
        + aux_cap_helper__b0,r5,band_UNPAIRED__condX2 - X__b0,r5,band_UNPAIRED
         = 0
 c2303: aux_cap_helper__b0,r7,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b0,r7,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b0,r7,band_HIGH_PAIRED__Z1  = 1
 c2304: - aux_cap_helper__b5,r7,band_LOW_PAIRED__condX1 <= 0
 c2305: 0.159946187556766 sv_function_helpervar_b5,r6__Z2
        - sv_function_helpervar_b5,r6__condX2 <= 0
 c2306: - aux_cap_helper__b7,r2,band_HIGH_PAIRED__condX1 <= 0
 c2307: sv_function_helpervar_b9,r13__condX3
        + sv_function_helpervar_b9,r13__condX2
        + sv_function_helpervar_b9,r13__condX1 - aux_c_b9,r13  = 0
 c2308: aux_cap_helper__b4,r4,band_UNPAIRED__Z1
        + aux_cap_helper__b4,r4,band_UNPAIRED__Z2  = 1
 c2309: 3.735962531 aux_cap_helper__b5,r6,band_LOW_PAIRED__condX1
        + 5.2303475434 aux_cap_helper__b5,r6,band_LOW_PAIRED__condX2
        - aux_cap_b5,r6,band_LOW_PAIRED
        - 1.4943850124 aux_cap_helper__b5,r6,band_LOW_PAIRED__Z2  = 0
 c2310: 39.9210131269397 sv_function_helpervar_b1,r8__Z3
        + 19.2432010437561 sv_function_helpervar_b1,r8__Z2
        + 5.2531243699 sv_function_helpervar_b1,r8__condX2
        + 2.1546007451 sv_function_helpervar_b1,r8__condX3
        + 125.5635947355 sv_function_helpervar_b1,r8__condX1 - aux_sv_b1,r8
         = 0
 c2311: - aux_cap_helper__b3,r13,band_UNPAIRED__condX2
        + aux_cap_helper__b3,r13,band_UNPAIRED__Z2 <= 0
 c2312: - X__b6,r0,band_UNPAIRED + aux_cap_helper__b6,r0,band_UNPAIRED__condX2
        + aux_cap_helper__b6,r0,band_UNPAIRED__condX1  = 0
 c2313: aux_cap_helper__b0,r8,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b0,r8,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b0,r8,band_HIGH_PAIRED__condX1
        - X__b0,r8,band_HIGH_PAIRED  = 0
 c2314: 14 GlobalBidder_hat{W}__i8,k1 + GlobalBidder_W_i_8 <= 27
 c2315: 0.159946187460974 sv_function_helpervar_b2,r12__Z2
        - sv_function_helpervar_b2,r12__condX2 <= 0
 c2316: aux_cap_helper__b8,r9,band_LOW_PAIRED__Z2
        + aux_cap_helper__b8,r9,band_LOW_PAIRED__Z1  = 1
 c2317: - aux_Omega_b0,r11 + 2481.31994997737 aux_sv_b0,r11  = 0
 c2318: aux_cap_helper__b5,r7,band_UNPAIRED__condX1
        - aux_cap_helper__b5,r7,band_UNPAIRED__Z1 <= 0
 c2319: 5.2303475434 aux_cap_helper__b9,r3,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b9,r3,band_LOW_PAIRED__condX1
        - 1.4943850124 aux_cap_helper__b9,r3,band_LOW_PAIRED__Z2
        - aux_cap_b9,r3,band_LOW_PAIRED  = 0
 c2320: sv_function_helpervar_b2,r0__Z1 + sv_function_helpervar_b2,r0__Z2
        + sv_function_helpervar_b2,r0__Z3  = 1
 c2321: - 1.4943850124 aux_cap_helper__b3,r2,band_LOW_PAIRED__Z2
        + 5.2303475434 aux_cap_helper__b3,r2,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b3,r2,band_LOW_PAIRED__condX1
        - aux_cap_b3,r2,band_LOW_PAIRED  = 0
 c2322: - aux_cap_helper__b6,r6,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b6,r6,band_HIGH_PAIRED__Z3 <= 0
 c2323: - aux_Omega_b1,r6 + 65.1023737220739 aux_sv_b1,r6  = 0
 c2324: - aux_cap_b2,r3,band_UNPAIRED
        - 0.3214355447 aux_cap_helper__b2,r3,band_UNPAIRED__Z2
        + 1.1250244065 aux_cap_helper__b2,r3,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b2,r3,band_UNPAIRED__condX1  = 0
 c2325: - sv_function_helpervar_b9,r11__condX1 <= 0
 c2326: 0.8035888618 aux_cap_helper__b5,r5,band_UNPAIRED__condX1
        + 1.1250244065 aux_cap_helper__b5,r5,band_UNPAIRED__condX2
        - 0.3214355447 aux_cap_helper__b5,r5,band_UNPAIRED__Z2
        - aux_cap_b5,r5,band_UNPAIRED  = 0
 c2327: aux_cap_helper__b7,r8,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b7,r8,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b7,r8,band_HIGH_PAIRED__Z3  = 1
 c2328: - aux_cap_b2,r2,band_HIGH_PAIRED
        + 2.1635071398 aux_cap_helper__b2,r2,band_HIGH_PAIRED__condX2
        + 1.3908260184 aux_cap_helper__b2,r2,band_HIGH_PAIRED__condX3
        + 1.5453622427 aux_cap_helper__b2,r2,band_HIGH_PAIRED__condX1
        + 0.927217345653617 aux_cap_helper__b2,r2,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b2,r2,band_HIGH_PAIRED__Z2  = 0
 c2329: sv_function_helpervar_b1,r1__condX2
        + sv_function_helpervar_b1,r1__condX1
        + sv_function_helpervar_b1,r1__condX3 - aux_c_b1,r1  = 0
 c2330: - aux_cap_b4,r4,band_LOW_PAIRED
        + 3.735962531 aux_cap_helper__b4,r4,band_LOW_PAIRED__condX1
        + 5.2303475434 aux_cap_helper__b4,r4,band_LOW_PAIRED__condX2
        - 1.4943850124 aux_cap_helper__b4,r4,band_LOW_PAIRED__Z2  = 0
 c2331: 0.8035888618 aux_cap_helper__b4,r8,band_UNPAIRED__condX1
        + 1.1250244065 aux_cap_helper__b4,r8,band_UNPAIRED__condX2
        - aux_cap_b4,r8,band_UNPAIRED
        - 0.3214355447 aux_cap_helper__b4,r8,band_UNPAIRED__Z2  = 0
 c2332: - aux_cap_helper__b2,r5,band_UNPAIRED__Z1
        + aux_cap_helper__b2,r5,band_UNPAIRED__condX1 <= 0
 c2333: - aux_cap_helper__b6,r6,band_LOW_PAIRED__condX1 <= 0
 c2334: X__b2,r12,band_HIGH_PAIRED + X__b5,r12,band_HIGH_PAIRED
        + X__b7,r12,band_HIGH_PAIRED + X__b8,r12,band_HIGH_PAIRED
        + X__b4,r12,band_HIGH_PAIRED + X__b0,r12,band_HIGH_PAIRED
        + X__b3,r12,band_HIGH_PAIRED + X__b1,r12,band_HIGH_PAIRED
        + X__b6,r12,band_HIGH_PAIRED + X__b9,r12,band_HIGH_PAIRED <= 3
 c2335: aux_cap_helper__b7,r0,band_LOW_PAIRED__Z1
        + aux_cap_helper__b7,r0,band_LOW_PAIRED__Z2  = 1
 c2336: - aux_cap_helper__b4,r2,band_LOW_PAIRED__condX1 <= 0
 c2337: - aux_cap_helper__b4,r9,band_HIGH_PAIRED__condX1 <= 0
 c2338: - 6.67344018864281 sv_function_helpervar_b1,r8__Z2
        + sv_function_helpervar_b1,r8__condX2 <= 0
 c2339: - aux_cap_helper__b8,r10,band_UNPAIRED__Z1
        + aux_cap_helper__b8,r10,band_UNPAIRED__condX1 <= 0
 c2340: sv_function_helpervar_b1,r3__Z2 + sv_function_helpervar_b1,r3__Z1
        + sv_function_helpervar_b1,r3__Z3  = 1
 c2341: 5.2303475434 aux_cap_helper__b2,r0,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b2,r0,band_LOW_PAIRED__condX1
        - aux_cap_b2,r0,band_LOW_PAIRED
        - 1.4943850124 aux_cap_helper__b2,r0,band_LOW_PAIRED__Z2  = 0
 c2342: - aux_Omega_b3,r8 + 165.704524280548 aux_sv_b3,r8  = 0
 c2343: - 3 aux_cap_helper__b3,r3,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b3,r3,band_HIGH_PAIRED__condX3 <= 0
 c2344: - aux_cap_helper__b9,r12,band_UNPAIRED__condX1 <= 0
 c2345: - X__b4,r0,band_LOW_PAIRED
        + aux_cap_helper__b4,r0,band_LOW_PAIRED__condX1
        + aux_cap_helper__b4,r0,band_LOW_PAIRED__condX2  = 0
 c2346: - 3 aux_cap_helper__b6,r3,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b6,r3,band_HIGH_PAIRED__condX3 <= 0
 c2347: aux_cap_helper__b7,r8,band_UNPAIRED__Z2
        + aux_cap_helper__b7,r8,band_UNPAIRED__Z1  = 1
 c2348: - X__b6,r12,band_HIGH_PAIRED
        + aux_cap_helper__b6,r12,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b6,r12,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b6,r12,band_HIGH_PAIRED__condX3  = 0
 c2349: aux_cap_helper__b5,r10,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b5,r10,band_HIGH_PAIRED__condX2 <= 0
 c2350: aux_cap_helper__b0,r6,band_UNPAIRED__condX1
        - aux_cap_helper__b0,r6,band_UNPAIRED__Z1 <= 0
 c2351: - sv_function_helpervar_b6,r3__condX2
        + 0.159946187469737 sv_function_helpervar_b6,r3__Z2 <= 0
 c2352: sv_function_helpervar_b1,r6__condX2
        - 6.60983461492046 sv_function_helpervar_b1,r6__Z2 <= 0
 c2353: aux_cap_helper__b4,r13,band_UNPAIRED__condX2
        + aux_cap_helper__b4,r13,band_UNPAIRED__condX1
        - X__b4,r13,band_UNPAIRED  = 0
 c2354: aux_cap_helper__b7,r9,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b7,r9,band_HIGH_PAIRED__condX2 <= 0
 c2355: sv_function_helpervar_b3,r13__Z2 + sv_function_helpervar_b3,r13__Z3
        + sv_function_helpervar_b3,r13__Z1  = 1
 c2356: aux_cap_helper__b1,r2,band_UNPAIRED__condX1
        - aux_cap_helper__b1,r2,band_UNPAIRED__Z1 <= 0
 c2357: aux_cap_helper__b1,r3,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b1,r3,band_HIGH_PAIRED__Z2 <= 0
 c2358: aux_cap_helper__b0,r13,band_LOW_PAIRED__Z2
        - aux_cap_helper__b0,r13,band_LOW_PAIRED__condX2 <= 0
 c2359: - aux_cap_helper__b3,r1,band_UNPAIRED__Z1
        + aux_cap_helper__b3,r1,band_UNPAIRED__condX1 <= 0
 c2360: aux_cap_helper__b0,r7,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b0,r7,band_HIGH_PAIRED__Z1 <= 0
 c2361: - 3 aux_cap_helper__b8,r2,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b8,r2,band_HIGH_PAIRED__condX3 <= 0
 c2362: sv_function_helpervar_b9,r0__condX3
        - 15.9946187432626 sv_function_helpervar_b9,r0__Z3 <= 0
 c2363: - aux_c_b0,r10 + aux_cap_b0,r10,band_LOW_PAIRED
        + aux_cap_b0,r10,band_HIGH_PAIRED + aux_cap_b0,r10,band_UNPAIRED  = 0
 c2364: aux_cap_helper__b3,r10,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b3,r10,band_HIGH_PAIRED__Z3 <= 0
 c2365: aux_cap_helper__b6,r6,band_LOW_PAIRED__Z2
        - aux_cap_helper__b6,r6,band_LOW_PAIRED__condX2 <= 0
 c2366: 2.1635071398 aux_cap_helper__b9,r3,band_HIGH_PAIRED__condX2
        + 1.3908260184 aux_cap_helper__b9,r3,band_HIGH_PAIRED__condX3
        + 1.5453622427 aux_cap_helper__b9,r3,band_HIGH_PAIRED__condX1
        - aux_cap_b9,r3,band_HIGH_PAIRED
        - 0.6181448971 aux_cap_helper__b9,r3,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b9,r3,band_HIGH_PAIRED__Z3  = 0
 c2367: sv_function_helpervar_b3,r8__condX2
        - 5.93313076510357 sv_function_helpervar_b3,r8__Z2 <= 0
 c2368: - aux_cap_helper__b7,r9,band_HIGH_PAIRED__condX1 <= 0
 c2369: aux_cap_b8,r12,band_UNPAIRED - aux_c_b8,r12
        + aux_cap_b8,r12,band_HIGH_PAIRED + aux_cap_b8,r12,band_LOW_PAIRED  = 0
 c2370: - aux_cap_helper__b4,r9,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b4,r9,band_HIGH_PAIRED__Z2 <= 0
 c2371: - aux_cap_helper__b1,r9,band_UNPAIRED__Z1
        + aux_cap_helper__b1,r9,band_UNPAIRED__condX1 <= 0
 c2372: aux_cap_helper__b8,r8,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b8,r8,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b8,r8,band_HIGH_PAIRED__condX2
        - X__b8,r8,band_HIGH_PAIRED  = 0
 c2373: aux_cap_helper__b5,r8,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b5,r8,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b5,r8,band_HIGH_PAIRED__Z2  = 1
 c2374: aux_cap_helper__b4,r2,band_LOW_PAIRED__Z1
        + aux_cap_helper__b4,r2,band_LOW_PAIRED__Z2  = 1
 c2375: aux_cap_helper__b1,r10,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b1,r10,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b1,r10,band_HIGH_PAIRED__Z3  = 1
 c2376: - X__b9,r13,band_LOW_PAIRED
        + aux_cap_helper__b9,r13,band_LOW_PAIRED__condX2
        + aux_cap_helper__b9,r13,band_LOW_PAIRED__condX1  = 0
 c2377: aux_cap_helper__b9,r12,band_UNPAIRED__Z2
        - aux_cap_helper__b9,r12,band_UNPAIRED__condX2 <= 0
 c2378: aux_cap_helper__b2,r2,band_UNPAIRED__Z1
        + aux_cap_helper__b2,r2,band_UNPAIRED__Z2  = 1
 c2379: aux_cap_helper__b8,r7,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b8,r7,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b8,r7,band_HIGH_PAIRED__Z1  = 1
 c2380: 2 aux_cap_helper__b3,r6,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b3,r6,band_HIGH_PAIRED__condX3 <= 0
 c2381: 0.159946187443648 sv_function_helpervar_b7,r1__Z2
        - sv_function_helpervar_b7,r1__condX2 <= 0
 c2382: aux_cap_b6,r3,band_LOW_PAIRED - aux_c_b6,r3
        + aux_cap_b6,r3,band_HIGH_PAIRED + aux_cap_b6,r3,band_UNPAIRED  = 0
 c2383: sv_function_helpervar_b9,r11__condX2
        - 7.76584458655766 sv_function_helpervar_b9,r11__Z2 <= 0
 c2384: sv_function_helpervar_b3,r0__condX1
        + sv_function_helpervar_b3,r0__condX2
        + sv_function_helpervar_b3,r0__condX3 - aux_c_b3,r0  = 0
 c2385: - 2 aux_cap_helper__b7,r11,band_UNPAIRED__Z2
        + aux_cap_helper__b7,r11,band_UNPAIRED__condX2 <= 0
 c2386: aux_cap_helper__b0,r13,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b0,r13,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b0,r13,band_HIGH_PAIRED__Z3  = 1
 c2387: - 2 aux_cap_helper__b7,r3,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b7,r3,band_HIGH_PAIRED__condX2 <= 0
 c2388: - sv_function_helpervar_b9,r1__condX1 <= 0
 c2389: - X__b9,r2,band_UNPAIRED + aux_cap_helper__b9,r2,band_UNPAIRED__condX2
        + aux_cap_helper__b9,r2,band_UNPAIRED__condX1  = 0
 c2390: - aux_cap_helper__b1,r9,band_HIGH_PAIRED__condX1 <= 0
 c2391: aux_cap_b5,r5,band_HIGH_PAIRED - aux_c_b5,r5
        + aux_cap_b5,r5,band_LOW_PAIRED + aux_cap_b5,r5,band_UNPAIRED  = 0
 c2392: - 0.6181448971 aux_cap_helper__b5,r9,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b5,r9,band_HIGH_PAIRED__Z3
        - aux_cap_b5,r9,band_HIGH_PAIRED
        + 1.3908260184 aux_cap_helper__b5,r9,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b5,r9,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b5,r9,band_HIGH_PAIRED__condX1  = 0
 c2393: aux_cap_helper__b5,r2,band_LOW_PAIRED__Z2
        + aux_cap_helper__b5,r2,band_LOW_PAIRED__Z1  = 1
 c2394: 2 aux_cap_helper__b2,r12,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b2,r12,band_HIGH_PAIRED__condX3 <= 0
 c2395: - aux_cap_helper__b6,r3,band_UNPAIRED__Z1
        + aux_cap_helper__b6,r3,band_UNPAIRED__condX1 <= 0
 c2396: aux_cap_helper__b6,r7,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b6,r7,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b6,r7,band_HIGH_PAIRED__Z2  = 1
 c2397: aux_cap_helper__b4,r3,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b4,r3,band_HIGH_PAIRED__Z2 <= 0
 c2398: - aux_cap_b6,r2,band_UNPAIRED
        - 0.3214355447 aux_cap_helper__b6,r2,band_UNPAIRED__Z2
        + 0.8035888618 aux_cap_helper__b6,r2,band_UNPAIRED__condX1
        + 1.1250244065 aux_cap_helper__b6,r2,band_UNPAIRED__condX2  = 0
 c2399: 0.8035888618 aux_cap_helper__b3,r0,band_UNPAIRED__condX1
        + 1.1250244065 aux_cap_helper__b3,r0,band_UNPAIRED__condX2
        - aux_cap_b3,r0,band_UNPAIRED
        - 0.3214355447 aux_cap_helper__b3,r0,band_UNPAIRED__Z2  = 0
 c2400: X__b3,r0,band_HIGH_PAIRED + X__b4,r0,band_HIGH_PAIRED
        + X__b6,r0,band_HIGH_PAIRED + X__b9,r0,band_HIGH_PAIRED
        + X__b8,r0,band_HIGH_PAIRED + X__b7,r0,band_HIGH_PAIRED
        + X__b1,r0,band_HIGH_PAIRED + X__b5,r0,band_HIGH_PAIRED
        + X__b2,r0,band_HIGH_PAIRED + X__b0,r0,band_HIGH_PAIRED <= 3
 c2401: sv_function_helpervar_b7,r12__Z3 + sv_function_helpervar_b7,r12__Z1
        + sv_function_helpervar_b7,r12__Z2  = 1
 c2402: - aux_c_b8,r9 + sv_function_helpervar_b8,r9__condX1
        + sv_function_helpervar_b8,r9__condX2
        + sv_function_helpervar_b8,r9__condX3  = 0
 c2403: - aux_cap_helper__b9,r6,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b9,r6,band_HIGH_PAIRED__Z3 <= 0
 c2404: sv_function_helpervar_b6,r2__condX1
        + sv_function_helpervar_b6,r2__condX2
        + sv_function_helpervar_b6,r2__condX3 - aux_c_b6,r2  = 0
 c2405: - sv_function_helpervar_b6,r3__condX1 <= 0
 c2406: 0.0102040816326531 X__b7,r8,band_UNPAIRED - GlobalBidder_W_ir__b7,r8
        + 0.0102040816326531 X__b7,r8,band_HIGH_PAIRED
        + 0.0102040816326531 X__b7,r8,band_LOW_PAIRED <= 0
 c2407: sv_function_helpervar_b1,r9__condX3
        + sv_function_helpervar_b1,r9__condX1
        + sv_function_helpervar_b1,r9__condX2 - aux_c_b1,r9  = 0
 c2408: aux_cap_helper__b8,r11,band_LOW_PAIRED__condX1
        - X__b8,r11,band_LOW_PAIRED
        + aux_cap_helper__b8,r11,band_LOW_PAIRED__condX2  = 0
 c2409: 0.0102040816326531 X__b9,r4,band_UNPAIRED
        + 0.0102040816326531 X__b9,r4,band_HIGH_PAIRED
        - GlobalBidder_W_ir__b9,r4
        + 0.0102040816326531 X__b9,r4,band_LOW_PAIRED <= 0
 c2410: - X__b6,r1,band_HIGH_PAIRED
        + aux_cap_helper__b6,r1,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b6,r1,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b6,r1,band_HIGH_PAIRED__condX2  = 0
 c2411: 0.0102040816326531 X__b8,r6,band_HIGH_PAIRED
        + 0.0102040816326531 X__b8,r6,band_UNPAIRED - GlobalBidder_W_ir__b8,r6
        + 0.0102040816326531 X__b8,r6,band_LOW_PAIRED <= 0
 c2412: aux_cap_helper__b3,r7,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b3,r7,band_HIGH_PAIRED__Z1 <= 0
 c2413: - aux_cap_helper__b0,r1,band_HIGH_PAIRED__condX1 <= 0
 c2414: aux_cap_helper__b2,r10,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b2,r10,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b2,r10,band_HIGH_PAIRED__condX1
        - X__b2,r10,band_HIGH_PAIRED  = 0
 c2415: 0.159946187518603 sv_function_helpervar_b5,r3__Z2
        - sv_function_helpervar_b5,r3__condX2 <= 0
 c2416: aux_cap_helper__b7,r3,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b7,r3,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b7,r3,band_HIGH_PAIRED__condX1
        - X__b7,r3,band_HIGH_PAIRED  = 0
 c2417: aux_cap_helper__b6,r12,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b6,r12,band_HIGH_PAIRED__Z2 <= 0
 c2418: - aux_cap_helper__b5,r0,band_UNPAIRED__Z1
        + aux_cap_helper__b5,r0,band_UNPAIRED__condX1 <= 0
 c2419: - aux_cap_helper__b8,r5,band_LOW_PAIRED__Z1
        + aux_cap_helper__b8,r5,band_LOW_PAIRED__condX1 <= 0
 c2420: - aux_cap_helper__b9,r1,band_HIGH_PAIRED__condX1 <= 0
 c2421: aux_cap_helper__b0,r3,band_UNPAIRED__Z2
        - aux_cap_helper__b0,r3,band_UNPAIRED__condX2 <= 0
 c2422: - aux_cap_helper__b6,r1,band_LOW_PAIRED__Z1
        + aux_cap_helper__b6,r1,band_LOW_PAIRED__condX1 <= 0
 c2423: - aux_cap_helper__b0,r13,band_LOW_PAIRED__condX1 <= 0
 c2424: 0.159946187471245 sv_function_helpervar_b6,r8__Z2
        - sv_function_helpervar_b6,r8__condX2 <= 0
 c2425: aux_cap_helper__b7,r3,band_LOW_PAIRED__condX1
        - aux_cap_helper__b7,r3,band_LOW_PAIRED__Z1 <= 0
 c2426: - 15.9946187432626 sv_function_helpervar_b9,r9__Z3
        + sv_function_helpervar_b9,r9__condX3 <= 0
 c2427: - aux_cap_helper__b9,r7,band_LOW_PAIRED__Z1
        + aux_cap_helper__b9,r7,band_LOW_PAIRED__condX1 <= 0
 c2428: sv_function_helpervar_b5,r6__Z1 + sv_function_helpervar_b5,r6__Z2
        + sv_function_helpervar_b5,r6__Z3  = 1
 c2429: 1.3908260184 aux_cap_helper__b0,r5,band_HIGH_PAIRED__condX3
        + 1.5453622427 aux_cap_helper__b0,r5,band_HIGH_PAIRED__condX1
        + 2.1635071398 aux_cap_helper__b0,r5,band_HIGH_PAIRED__condX2
        - aux_cap_b0,r5,band_HIGH_PAIRED
        - 0.6181448971 aux_cap_helper__b0,r5,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b0,r5,band_HIGH_PAIRED__Z3  = 0
 c2430: - 2 aux_cap_helper__b1,r10,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b1,r10,band_HIGH_PAIRED__condX2 <= 0
 c2431: 0.159946187454416 sv_function_helpervar_b2,r9__Z2
        - sv_function_helpervar_b2,r9__condX2 <= 0
 c2432: - aux_cap_helper__b3,r1,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b3,r1,band_HIGH_PAIRED__Z2 <= 0
 c2433: aux_cap_helper__b3,r3,band_UNPAIRED__Z1
        + aux_cap_helper__b3,r3,band_UNPAIRED__Z2  = 1
 c2434: 0.159946187462481 sv_function_helpervar_b0,r8__Z2
        - sv_function_helpervar_b0,r8__condX2 <= 0
 c2435: - aux_cap_helper__b3,r13,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b3,r13,band_HIGH_PAIRED__Z3 <= 0
 c2436: - aux_Omega_b7,r7 + 4196.54466091694 aux_sv_b7,r7  = 0
 c2437: aux_cap_helper__b8,r10,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b8,r10,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b8,r10,band_HIGH_PAIRED__Z1  = 1
 c2438: aux_cap_helper__b3,r8,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b3,r8,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b3,r8,band_HIGH_PAIRED__Z1  = 1
 c2439: 0.159946187525574 sv_function_helpervar_b0,r1__Z2
        - sv_function_helpervar_b0,r1__condX2 <= 0
 c2440: aux_cap_b4,r7,band_HIGH_PAIRED - aux_c_b4,r7
        + aux_cap_b4,r7,band_LOW_PAIRED + aux_cap_b4,r7,band_UNPAIRED  = 0
 c2441: - aux_cap_helper__b0,r3,band_UNPAIRED__condX1 <= 0
 c2442: 1799.69578039488 aux_sv_b5,r11 - aux_Omega_b5,r11  = 0
 c2443: aux_cap_helper__b0,r1,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b0,r1,band_HIGH_PAIRED__condX2 <= 0
 c2444: sv_function_helpervar_b1,r2__condX3
        - 15.9946187432626 sv_function_helpervar_b1,r2__Z3 <= 0
 c2445: - sv_function_helpervar_b8,r11__condX3
        + 7.71565157693315 sv_function_helpervar_b8,r11__Z3 <= 0
 c2446: aux_cap_helper__b1,r12,band_LOW_PAIRED__Z1
        + aux_cap_helper__b1,r12,band_LOW_PAIRED__Z2  = 1
 c2447: - sv_function_helpervar_b1,r3__condX1 <= 0
 c2448: sv_function_helpervar_b1,r7__Z2 + sv_function_helpervar_b1,r7__Z1
        + sv_function_helpervar_b1,r7__Z3  = 1
 c2449: aux_cap_helper__b0,r2,band_UNPAIRED__condX1
        + aux_cap_helper__b0,r2,band_UNPAIRED__condX2 - X__b0,r2,band_UNPAIRED
         = 0
 c2450: - 2 aux_cap_helper__b1,r1,band_LOW_PAIRED__Z2
        + aux_cap_helper__b1,r1,band_LOW_PAIRED__condX2 <= 0
 c2451: - 2 aux_cap_helper__b2,r3,band_LOW_PAIRED__Z2
        + aux_cap_helper__b2,r3,band_LOW_PAIRED__condX2 <= 0
 c2452: 0.0102040816326531 X__b7,r11,band_UNPAIRED
        + 0.0102040816326531 X__b7,r11,band_LOW_PAIRED
        + 0.0102040816326531 X__b7,r11,band_HIGH_PAIRED
        - GlobalBidder_W_ir__b7,r11 <= 0
 c2453: sv_function_helpervar_b8,r1__condX2
        + sv_function_helpervar_b8,r1__condX3
        + sv_function_helpervar_b8,r1__condX1 - aux_c_b8,r1  = 0
 c2454: aux_cap_helper__b4,r2,band_LOW_PAIRED__Z2
        - aux_cap_helper__b4,r2,band_LOW_PAIRED__condX2 <= 0
 c2455: sv_function_helpervar_b9,r1__Z3 + sv_function_helpervar_b9,r1__Z1
        + sv_function_helpervar_b9,r1__Z2  = 1
 c2456: aux_cap_helper__b5,r4,band_LOW_PAIRED__Z2
        - aux_cap_helper__b5,r4,band_LOW_PAIRED__condX2 <= 0
 c2457: 1719.094587965 sv_function_helpervar_b9,r0__condX1
        + 64.9900375872 sv_function_helpervar_b9,r0__condX2
        + 31.8739219455 sv_function_helpervar_b9,r0__condX3
        + 264.567716498342 sv_function_helpervar_b9,r0__Z2
        + 508.568864087871 sv_function_helpervar_b9,r0__Z3 - aux_sv_b9,r0  = 0
 c2458: aux_cap_helper__b9,r13,band_LOW_PAIRED__Z2
        + aux_cap_helper__b9,r13,band_LOW_PAIRED__Z1  = 1
 c2459: - 0.159946187434666 sv_function_helpervar_b7,r8__Z1
        + sv_function_helpervar_b7,r8__condX1 <= 0
 c2460: - 3 aux_cap_helper__b8,r12,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b8,r12,band_HIGH_PAIRED__condX3 <= 0
 c2461: sv_function_helpervar_b9,r10__condX3
        - 15.9946187432626 sv_function_helpervar_b9,r10__Z3 <= 0
 c2462: - aux_cap_helper__b9,r11,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b9,r11,band_HIGH_PAIRED__Z2 <= 0
 c2463: aux_cap_helper__b9,r9,band_LOW_PAIRED__Z2
        + aux_cap_helper__b9,r9,band_LOW_PAIRED__Z1  = 1
 c2464: - aux_sv_b1,r12 + 6.2277482191 sv_function_helpervar_b1,r12__condX2
        + 125.5635947294 sv_function_helpervar_b1,r12__condX1
        + 1.9422056904 sv_function_helpervar_b1,r12__condX3
        + 43.3181910498652 sv_function_helpervar_b1,r12__Z3
        + 19.0873136748382 sv_function_helpervar_b1,r12__Z2  = 0
 c2465: 0.159946187479682 sv_function_helpervar_b2,r5__Z2
        - sv_function_helpervar_b2,r5__condX2 <= 0
 c2466: aux_cap_helper__b3,r8,band_UNPAIRED__condX1
        - aux_cap_helper__b3,r8,band_UNPAIRED__Z1 <= 0
 c2467: aux_cap_b3,r9,band_UNPAIRED + aux_cap_b3,r9,band_LOW_PAIRED
        - aux_c_b3,r9 + aux_cap_b3,r9,band_HIGH_PAIRED  = 0
 c2468: aux_cap_helper__b8,r0,band_LOW_PAIRED__Z2
        + aux_cap_helper__b8,r0,band_LOW_PAIRED__Z1  = 1
 c2469: 6.16837738919705 sv_function_helpervar_b0,r8__Z3
        - sv_function_helpervar_b0,r8__condX3 <= 0
 c2470: aux_cap_helper__b1,r4,band_LOW_PAIRED__Z2
        + aux_cap_helper__b1,r4,band_LOW_PAIRED__Z1  = 1
 c2471: - 0.159946187439412 sv_function_helpervar_b2,r6__Z1
        + sv_function_helpervar_b2,r6__condX1 <= 0
 c2472: sv_function_helpervar_b6,r2__condX3
        - 15.9946187432626 sv_function_helpervar_b6,r2__Z3 <= 0
 c2473: aux_cap_helper__b3,r1,band_UNPAIRED__condX2
        + aux_cap_helper__b3,r1,band_UNPAIRED__condX1 - X__b3,r1,band_UNPAIRED
         = 0
 c2474: - aux_cap_helper__b9,r11,band_HIGH_PAIRED__condX1 <= 0
 c2475: aux_cap_helper__b4,r4,band_UNPAIRED__condX1
        - aux_cap_helper__b4,r4,band_UNPAIRED__Z1 <= 0
 c2476: aux_cap_helper__b4,r10,band_UNPAIRED__condX2
        + aux_cap_helper__b4,r10,band_UNPAIRED__condX1
        - X__b4,r10,band_UNPAIRED  = 0
 c2477: - aux_sv_b4,r13 + 20.7071061071 sv_function_helpervar_b4,r13__condX3
        + 1289.3272631547 sv_function_helpervar_b4,r13__condX1
        + 59.7966412981 sv_function_helpervar_b4,r13__condX2
        + 196.658735361317 sv_function_helpervar_b4,r13__Z2
        + 432.586547953621 sv_function_helpervar_b4,r13__Z3  = 0
 c2478: sv_function_helpervar_b8,r9__Z3 + sv_function_helpervar_b8,r9__Z2
        + sv_function_helpervar_b8,r9__Z1  = 1
 c2479: aux_cap_b1,r6,band_UNPAIRED + aux_cap_b1,r6,band_HIGH_PAIRED
        - aux_c_b1,r6 + aux_cap_b1,r6,band_LOW_PAIRED  = 0
 c2480: - aux_c_b4,r10 + sv_function_helpervar_b4,r10__condX1
        + sv_function_helpervar_b4,r10__condX3
        + sv_function_helpervar_b4,r10__condX2  = 0
 c2481: sv_function_helpervar_b3,r8__Z1 + sv_function_helpervar_b3,r8__Z3
        + sv_function_helpervar_b3,r8__Z2  = 1
 c2482: - 2 aux_cap_helper__b7,r11,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b7,r11,band_HIGH_PAIRED__condX2 <= 0
 c2483: sv_function_helpervar_b9,r0__condX1
        + sv_function_helpervar_b9,r0__condX2
        + sv_function_helpervar_b9,r0__condX3 - aux_c_b9,r0  = 0
 c2484: - aux_cap_helper__b9,r5,band_LOW_PAIRED__condX2
        + aux_cap_helper__b9,r5,band_LOW_PAIRED__Z2 <= 0
 c2485: - aux_cap_helper__b5,r13,band_LOW_PAIRED__condX1 <= 0
 c2486: aux_cap_helper__b6,r1,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b6,r1,band_HIGH_PAIRED__condX2 <= 0
 c2487: aux_cap_helper__b4,r10,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b4,r10,band_HIGH_PAIRED__condX2 <= 0
 c2488: - 0.159946187476732 sv_function_helpervar_b8,r1__Z1
        + sv_function_helpervar_b8,r1__condX1 <= 0
 c2489: - aux_Omega_b4,r4 + 1489.40086221226 aux_sv_b4,r4  = 0
 c2490: - aux_cap_b1,r1,band_LOW_PAIRED
        - 1.4943850124 aux_cap_helper__b1,r1,band_LOW_PAIRED__Z2
        + 5.2303475434 aux_cap_helper__b1,r1,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b1,r1,band_LOW_PAIRED__condX1  = 0
 c2491: - aux_sv_b4,r12 + 195.608255576622 sv_function_helpervar_b4,r12__Z2
        + 450.857639267 sv_function_helpervar_b4,r12__Z3
        + 66.3643488747 sv_function_helpervar_b4,r12__condX2
        + 19.5647787027 sv_function_helpervar_b4,r12__condX3
        + 1289.3272634824 sv_function_helpervar_b4,r12__condX1  = 0
 c2492: sv_function_helpervar_b5,r0__Z3 + sv_function_helpervar_b5,r0__Z2
        + sv_function_helpervar_b5,r0__Z1  = 1
 c2493: - aux_c_b2,r1 + sv_function_helpervar_b2,r1__condX1
        + sv_function_helpervar_b2,r1__condX3
        + sv_function_helpervar_b2,r1__condX2  = 0
 c2494: sv_function_helpervar_b5,r8__condX1
        - 0.159946187453478 sv_function_helpervar_b5,r8__Z1 <= 0
 c2495: - aux_cap_helper__b2,r12,band_UNPAIRED__condX1 <= 0
 c2496: - aux_cap_helper__b3,r5,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b3,r5,band_HIGH_PAIRED__Z2 <= 0
 c2497: X__b5,r11,band_HIGH_PAIRED + X__b2,r11,band_HIGH_PAIRED
        + X__b0,r11,band_HIGH_PAIRED + X__b4,r11,band_HIGH_PAIRED
        + X__b3,r11,band_HIGH_PAIRED + X__b8,r11,band_HIGH_PAIRED
        + X__b6,r11,band_HIGH_PAIRED + X__b7,r11,band_HIGH_PAIRED
        + X__b9,r11,band_HIGH_PAIRED + X__b1,r11,band_HIGH_PAIRED <= 3
 c2498: - aux_Omega_b0,r10 + 334.534208659939 aux_sv_b0,r10  = 0
 c2499: aux_cap_helper__b1,r1,band_UNPAIRED__condX1
        + aux_cap_helper__b1,r1,band_UNPAIRED__condX2 - X__b1,r1,band_UNPAIRED
         = 0
 c2500: sv_function_helpervar_b9,r11__Z1 + sv_function_helpervar_b9,r11__Z2
        + sv_function_helpervar_b9,r11__Z3  = 1
 c2501: - X__b4,r13,band_LOW_PAIRED
        + aux_cap_helper__b4,r13,band_LOW_PAIRED__condX1
        + aux_cap_helper__b4,r13,band_LOW_PAIRED__condX2  = 0
 c2502: aux_cap_b0,r0,band_LOW_PAIRED + aux_cap_b0,r0,band_UNPAIRED
        - aux_c_b0,r0 + aux_cap_b0,r0,band_HIGH_PAIRED  = 0
 c2503: sv_function_helpervar_b4,r4__Z1 + sv_function_helpervar_b4,r4__Z2
        + sv_function_helpervar_b4,r4__Z3  = 1
 c2504: - aux_cap_helper__b7,r1,band_UNPAIRED__condX1 <= 0
 c2505: - aux_cap_helper__b8,r7,band_HIGH_PAIRED__condX1 <= 0
 c2506: aux_cap_helper__b9,r4,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b9,r4,band_HIGH_PAIRED__Z3 <= 0
 c2507: - 15.9946187432626 sv_function_helpervar_b4,r6__Z3
        + sv_function_helpervar_b4,r6__condX3 <= 0
 c2508: - aux_cap_helper__b7,r11,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b7,r11,band_HIGH_PAIRED__condX1 <= 0
 c2509: aux_cap_helper__b1,r3,band_LOW_PAIRED__condX1
        - aux_cap_helper__b1,r3,band_LOW_PAIRED__Z1 <= 0
 c2510: - 0.159946187445632 sv_function_helpervar_b3,r12__Z1
        + sv_function_helpervar_b3,r12__condX1 <= 0
 c2511: aux_cap_helper__b4,r8,band_LOW_PAIRED__Z1
        + aux_cap_helper__b4,r8,band_LOW_PAIRED__Z2  = 1
 c2512: aux_cap_helper__b6,r8,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b6,r8,band_LOW_PAIRED__Z2 <= 0
 c2513: - aux_cap_helper__b2,r2,band_UNPAIRED__condX1 <= 0
 c2514: - 2 aux_cap_helper__b6,r9,band_UNPAIRED__Z2
        + aux_cap_helper__b6,r9,band_UNPAIRED__condX2 <= 0
 c2515: - aux_cap_helper__b9,r13,band_LOW_PAIRED__Z1
        + aux_cap_helper__b9,r13,band_LOW_PAIRED__condX1 <= 0
 c2516: - aux_cap_helper__b7,r7,band_LOW_PAIRED__condX1 <= 0
 c2517: 0.159946187460821 sv_function_helpervar_b3,r4__Z2
        - sv_function_helpervar_b3,r4__condX2 <= 0
 c2518: - X__b7,r0,band_HIGH_PAIRED
        + aux_cap_helper__b7,r0,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b7,r0,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b7,r0,band_HIGH_PAIRED__condX1  = 0
 c2519: - 0.3214355447 aux_cap_helper__b8,r5,band_UNPAIRED__Z2
        + 1.1250244065 aux_cap_helper__b8,r5,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b8,r5,band_UNPAIRED__condX1
        - aux_cap_b8,r5,band_UNPAIRED  = 0
 c2520: sv_function_helpervar_b8,r7__condX1
        - 0.159946187436067 sv_function_helpervar_b8,r7__Z1 <= 0
 c2521: - aux_cap_b2,r1,band_UNPAIRED
        - 0.3214355447 aux_cap_helper__b2,r1,band_UNPAIRED__Z2
        + 0.8035888618 aux_cap_helper__b2,r1,band_UNPAIRED__condX1
        + 1.1250244065 aux_cap_helper__b2,r1,band_UNPAIRED__condX2  = 0
 c2522: sv_function_helpervar_b5,r4__condX3
        - 15.9946187432626 sv_function_helpervar_b5,r4__Z3 <= 0
 c2523: aux_cap_helper__b0,r4,band_LOW_PAIRED__Z2
        - aux_cap_helper__b0,r4,band_LOW_PAIRED__condX2 <= 0
 c2524: 1.1250244065 aux_cap_helper__b5,r3,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b5,r3,band_UNPAIRED__condX1
        - 0.3214355447 aux_cap_helper__b5,r3,band_UNPAIRED__Z2
        - aux_cap_b5,r3,band_UNPAIRED  = 0
 c2525: 5.85581773625452 sv_function_helpervar_b2,r7__Z3
        - sv_function_helpervar_b2,r7__condX3 <= 0
 c2526: - sv_function_helpervar_b3,r13__condX1 <= 0
 c2527: - aux_cap_helper__b8,r6,band_UNPAIRED__condX2
        + aux_cap_helper__b8,r6,band_UNPAIRED__Z2 <= 0
 c2528: aux_cap_helper__b9,r2,band_LOW_PAIRED__Z2
        + aux_cap_helper__b9,r2,band_LOW_PAIRED__Z1  = 1
 c2529: - 2 aux_cap_helper__b9,r13,band_LOW_PAIRED__Z2
        + aux_cap_helper__b9,r13,band_LOW_PAIRED__condX2 <= 0
 c2530: aux_cap_helper__b0,r0,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b0,r0,band_UNPAIRED__Z2 <= 0
 c2531: aux_cap_helper__b2,r2,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b2,r2,band_HIGH_PAIRED__Z2 <= 0
 c2532: aux_cap_helper__b5,r7,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b5,r7,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b5,r7,band_HIGH_PAIRED__Z3  = 1
 c2533: - aux_cap_helper__b2,r13,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b2,r13,band_HIGH_PAIRED__condX1 <= 0
 c2534: - X__b0,r9,band_LOW_PAIRED
        + aux_cap_helper__b0,r9,band_LOW_PAIRED__condX1
        + aux_cap_helper__b0,r9,band_LOW_PAIRED__condX2  = 0
 c2535: sv_function_helpervar_b6,r1__condX2
        - 5.8772093296743 sv_function_helpervar_b6,r1__Z2 <= 0
 c2536: 1.1250244065 aux_cap_helper__b0,r6,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b0,r6,band_UNPAIRED__condX1
        - 0.3214355447 aux_cap_helper__b0,r6,band_UNPAIRED__Z2
        - aux_cap_b0,r6,band_UNPAIRED  = 0
 c2537: 2 aux_cap_helper__b6,r1,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b6,r1,band_HIGH_PAIRED__condX3 <= 0
 c2538: aux_cap_helper__b9,r9,band_UNPAIRED__condX1
        + aux_cap_helper__b9,r9,band_UNPAIRED__condX2 - X__b9,r9,band_UNPAIRED
         = 0
 c2539: aux_cap_helper__b6,r8,band_UNPAIRED__condX1
        + aux_cap_helper__b6,r8,band_UNPAIRED__condX2 - X__b6,r8,band_UNPAIRED
         = 0
 c2540: aux_cap_helper__b0,r10,band_LOW_PAIRED__Z2
        - aux_cap_helper__b0,r10,band_LOW_PAIRED__condX2 <= 0
 c2541: 1.1250244065 aux_cap_helper__b3,r8,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b3,r8,band_UNPAIRED__condX1
        - 0.3214355447 aux_cap_helper__b3,r8,band_UNPAIRED__Z2
        - aux_cap_b3,r8,band_UNPAIRED  = 0
 c2542: 2 aux_cap_helper__b4,r10,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b4,r10,band_HIGH_PAIRED__condX3 <= 0
 c2543: sv_function_helpervar_b9,r6__condX2
        - 6.20001223735563 sv_function_helpervar_b9,r6__Z2 <= 0
 c2544: - X__b4,r4,band_HIGH_PAIRED
        + aux_cap_helper__b4,r4,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b4,r4,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b4,r4,band_HIGH_PAIRED__condX3  = 0
 c2545: - aux_Omega_b5,r2 + 187.732322409016 aux_sv_b5,r2  = 0
 c2546: 2 aux_cap_helper__b0,r9,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b0,r9,band_HIGH_PAIRED__condX3 <= 0
 c2547: sv_function_helpervar_b9,r5__Z3 + sv_function_helpervar_b9,r5__Z1
        + sv_function_helpervar_b9,r5__Z2  = 1
 c2548: - aux_cap_helper__b6,r9,band_UNPAIRED__Z1
        + aux_cap_helper__b6,r9,band_UNPAIRED__condX1 <= 0
 c2549: - aux_cap_helper__b6,r12,band_LOW_PAIRED__condX1 <= 0
 c2550: - aux_cap_b6,r12,band_UNPAIRED
        - 0.3214355447 aux_cap_helper__b6,r12,band_UNPAIRED__Z2
        + 1.1250244065 aux_cap_helper__b6,r12,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b6,r12,band_UNPAIRED__condX1  = 0
 c2551: - 0.15994618748582 sv_function_helpervar_b2,r7__Z1
        + sv_function_helpervar_b2,r7__condX1 <= 0
 c2552: - X__b6,r5,band_HIGH_PAIRED
        + aux_cap_helper__b6,r5,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b6,r5,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b6,r5,band_HIGH_PAIRED__condX3  = 0
 c2553: aux_cap_b3,r3,band_UNPAIRED + aux_cap_b3,r3,band_LOW_PAIRED
        - aux_c_b3,r3 + aux_cap_b3,r3,band_HIGH_PAIRED  = 0
 c2554: - aux_cap_helper__b1,r13,band_UNPAIRED__condX2
        + aux_cap_helper__b1,r13,band_UNPAIRED__Z2 <= 0
 c2555: - aux_cap_helper__b4,r11,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b4,r11,band_HIGH_PAIRED__Z3 <= 0
 c2556: - aux_cap_helper__b3,r8,band_LOW_PAIRED__condX1 <= 0
 c2557: - X__b0,r12,band_HIGH_PAIRED
        + aux_cap_helper__b0,r12,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b0,r12,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b0,r12,band_HIGH_PAIRED__condX3  = 0
 c2558: - 3 aux_cap_helper__b9,r13,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b9,r13,band_HIGH_PAIRED__condX3 <= 0
 c2559: aux_cap_helper__b7,r10,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b7,r10,band_HIGH_PAIRED__Z1 <= 0
 c2560: - aux_cap_b8,r0,band_LOW_PAIRED
        + 5.2303475434 aux_cap_helper__b8,r0,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b8,r0,band_LOW_PAIRED__condX1
        - 1.4943850124 aux_cap_helper__b8,r0,band_LOW_PAIRED__Z2  = 0
 c2561: aux_cap_helper__b6,r7,band_UNPAIRED__Z2
        + aux_cap_helper__b6,r7,band_UNPAIRED__Z1  = 1
 c2562: aux_cap_helper__b9,r3,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b9,r3,band_HIGH_PAIRED__Z3 <= 0
 c2563: aux_cap_helper__b9,r0,band_UNPAIRED__condX1
        + aux_cap_helper__b9,r0,band_UNPAIRED__condX2 - X__b9,r0,band_UNPAIRED
         = 0
 c2564: - aux_cap_helper__b8,r1,band_UNPAIRED__condX1 <= 0
 c2565: 2 aux_cap_helper__b3,r4,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b3,r4,band_HIGH_PAIRED__condX3 <= 0
 c2566: - aux_cap_helper__b4,r3,band_LOW_PAIRED__condX2
        + aux_cap_helper__b4,r3,band_LOW_PAIRED__Z2 <= 0
 c2567: - aux_Omega_b4,r1 + 161.483462587009 aux_sv_b4,r1  = 0
 c2568: aux_cap_helper__b0,r9,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b0,r9,band_HIGH_PAIRED__condX2 <= 0
 c2569: aux_cap_helper__b2,r1,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b2,r1,band_HIGH_PAIRED__Z1 <= 0
 c2570: - X__b4,r2,band_UNPAIRED + aux_cap_helper__b4,r2,band_UNPAIRED__condX1
        + aux_cap_helper__b4,r2,band_UNPAIRED__condX2  = 0
 c2571: X__b8,r10,band_HIGH_PAIRED + X__b8,r10,band_LOW_PAIRED
        - GlobalBidder_W_ir__b8,r10 + X__b8,r10,band_UNPAIRED >= 0
 c2572: 61.2824318032915 aux_sv_b8,r13 - aux_Omega_b8,r13  = 0
 c2573: aux_cap_helper__b8,r13,band_UNPAIRED__condX1
        - aux_cap_helper__b8,r13,band_UNPAIRED__Z1 <= 0
 c2574: aux_cap_helper__b3,r1,band_UNPAIRED__Z2
        + aux_cap_helper__b3,r1,band_UNPAIRED__Z1  = 1
 c2575: aux_cap_helper__b9,r9,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b9,r9,band_HIGH_PAIRED__Z1 <= 0
 c2576: sv_function_helpervar_b7,r13__condX3
        - 15.9946187432626 sv_function_helpervar_b7,r13__Z3 <= 0
 c2577: 35.8037783276 sv_function_helpervar_b9,r13__condX3
        + 57.4441421382 sv_function_helpervar_b9,r13__condX2
        + 1719.0945881454 sv_function_helpervar_b9,r13__condX1
        + 265.774653707375 sv_function_helpervar_b9,r13__Z2
        + 445.71230953988 sv_function_helpervar_b9,r13__Z3 - aux_sv_b9,r13  = 0
 c2578: 315.555631104965 aux_sv_b0,r1 - aux_Omega_b0,r1  = 0
 c2579: aux_cap_helper__b5,r12,band_LOW_PAIRED__Z1
        + aux_cap_helper__b5,r12,band_LOW_PAIRED__Z2  = 1
 c2580: aux_cap_b4,r9,band_UNPAIRED + aux_cap_b4,r9,band_LOW_PAIRED
        + aux_cap_b4,r9,band_HIGH_PAIRED - aux_c_b4,r9  = 0
 c2581: 0.159946187462914 sv_function_helpervar_b0,r13__Z2
        - sv_function_helpervar_b0,r13__condX2 <= 0
 c2582: 2.6375238745 sv_function_helpervar_b2,r3__condX3
        + 7.2072524918 sv_function_helpervar_b2,r3__condX2
        + 160.8367693211 sv_function_helpervar_b2,r3__condX1 - aux_sv_b2,r3
        + 24.5724555094298 sv_function_helpervar_b2,r3__Z2
        + 53.0924336765699 sv_function_helpervar_b2,r3__Z3  = 0
 c2583: GlobalBidder_W_i_9 - 14 GlobalBidder_hat{W}__i9,k1 >= -1
 c2584: - aux_cap_helper__b4,r11,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b4,r11,band_HIGH_PAIRED__Z2 <= 0
 c2585: aux_cap_b4,r12,band_UNPAIRED + aux_cap_b4,r12,band_HIGH_PAIRED
        + aux_cap_b4,r12,band_LOW_PAIRED - aux_c_b4,r12  = 0
 c2586: aux_cap_helper__b8,r3,band_LOW_PAIRED__condX1
        + aux_cap_helper__b8,r3,band_LOW_PAIRED__condX2
        - X__b8,r3,band_LOW_PAIRED  = 0
 c2587: - aux_cap_helper__b6,r0,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b6,r0,band_HIGH_PAIRED__Z3 <= 0
 c2588: aux_cap_helper__b0,r0,band_UNPAIRED__condX1
        - aux_cap_helper__b0,r0,band_UNPAIRED__Z1 <= 0
 c2589: aux_cap_helper__b3,r6,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b3,r6,band_HIGH_PAIRED__condX2 <= 0
 c2590: aux_cap_helper__b4,r7,band_UNPAIRED__Z2
        - aux_cap_helper__b4,r7,band_UNPAIRED__condX2 <= 0
 c2591: - aux_cap_helper__b0,r7,band_LOW_PAIRED__condX2
        + aux_cap_helper__b0,r7,band_LOW_PAIRED__Z2 <= 0
 c2592: - X__b1,r10,band_LOW_PAIRED
        + aux_cap_helper__b1,r10,band_LOW_PAIRED__condX2
        + aux_cap_helper__b1,r10,band_LOW_PAIRED__condX1  = 0
 c2593: aux_cap_helper__b5,r9,band_LOW_PAIRED__Z2
        - aux_cap_helper__b5,r9,band_LOW_PAIRED__condX2 <= 0
 c2594: X__b9,r5,band_UNPAIRED + X__b2,r5,band_UNPAIRED
        + X__b8,r5,band_UNPAIRED + X__b7,r5,band_UNPAIRED
        + X__b3,r5,band_UNPAIRED + X__b5,r5,band_UNPAIRED
        + X__b6,r5,band_UNPAIRED + X__b4,r5,band_UNPAIRED
        + X__b1,r5,band_UNPAIRED + X__b0,r5,band_UNPAIRED <= 2
 c2595: - 3 aux_cap_helper__b6,r7,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b6,r7,band_HIGH_PAIRED__condX3 <= 0
 c2596: 19.1618858942406 sv_function_helpervar_b1,r0__Z2
        + 41.9225300868796 sv_function_helpervar_b1,r0__Z3 - aux_sv_b1,r0
        + 5.7615150407 sv_function_helpervar_b1,r0__condX2
        + 2.029463848 sv_function_helpervar_b1,r0__condX3
        + 125.563594738 sv_function_helpervar_b1,r0__condX1  = 0
 c2597: - X__b4,r6,band_LOW_PAIRED
        + aux_cap_helper__b4,r6,band_LOW_PAIRED__condX2
        + aux_cap_helper__b4,r6,band_LOW_PAIRED__condX1  = 0
 c2598: - aux_cap_helper__b7,r4,band_LOW_PAIRED__condX1 <= 0
 c2599: - 0.159946187486662 sv_function_helpervar_b6,r5__Z1
        + sv_function_helpervar_b6,r5__condX1 <= 0
 c2600: - 0.159946187451481 sv_function_helpervar_b4,r0__Z1
        + sv_function_helpervar_b4,r0__condX1 <= 0
 c2601: - aux_cap_helper__b2,r2,band_LOW_PAIRED__condX1 <= 0
 c2602: sv_function_helpervar_b1,r11__condX1
        - 0.159946187471509 sv_function_helpervar_b1,r11__Z1 <= 0
 c2603: sv_function_helpervar_b4,r13__condX3
        + sv_function_helpervar_b4,r13__condX1
        + sv_function_helpervar_b4,r13__condX2 - aux_c_b4,r13  = 0
 c2604: aux_cap_helper__b5,r9,band_UNPAIRED__condX1
        - aux_cap_helper__b5,r9,band_UNPAIRED__Z1 <= 0
 c2605: 1719.0945882811 sv_function_helpervar_b9,r10__condX1
        + 62.0544720697 sv_function_helpervar_b9,r10__condX2
        + 33.1856751511 sv_function_helpervar_b9,r10__condX3 - aux_sv_b9,r10
        + 487.587871678127 sv_function_helpervar_b9,r10__Z3
        + 265.037249012739 sv_function_helpervar_b9,r10__Z2  = 0
 c2606: sv_function_helpervar_b3,r8__condX3
        - 15.9946187432626 sv_function_helpervar_b3,r8__Z3 <= 0
 c2607: aux_cap_helper__b7,r10,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b7,r10,band_HIGH_PAIRED__Z2 <= 0
 c2608: - X__b9,r12,band_UNPAIRED
        + aux_cap_helper__b9,r12,band_UNPAIRED__condX1
        + aux_cap_helper__b9,r12,band_UNPAIRED__condX2  = 0
 c2609: - aux_cap_b2,r11,band_UNPAIRED
        + 1.1250244065 aux_cap_helper__b2,r11,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b2,r11,band_UNPAIRED__condX1
        - 0.3214355447 aux_cap_helper__b2,r11,band_UNPAIRED__Z2  = 0
 c2610: - aux_cap_helper__b6,r0,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b6,r0,band_HIGH_PAIRED__Z2 <= 0
 c2611: aux_cap_helper__b2,r1,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b2,r1,band_HIGH_PAIRED__Z2 <= 0
 c2612: aux_cap_helper__b3,r1,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b3,r1,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b3,r1,band_HIGH_PAIRED__Z2  = 1
 c2613: aux_Omega_b9,r1 + aux_Omega_b9,r0 + aux_Omega_b9,r3 + aux_Omega_b9,r2
        + aux_Omega_b9,r5 + aux_Omega_b9,r4 + aux_Omega_b9,r7 + aux_Omega_b9,r6
        + aux_Omega_b9,r9 + aux_Omega_b9,r8 - GlobalBidder_PSI__i9,k2
        + aux_Omega_b9,r13 + aux_Omega_b9,r10 + aux_Omega_b9,r11
        + aux_Omega_b9,r12 - 18497526.8940961 GlobalBidder_hat{W}__i9,k2
        >= -18497526.8940961
 c2614: - aux_cap_helper__b7,r6,band_UNPAIRED__condX2
        + aux_cap_helper__b7,r6,band_UNPAIRED__Z2 <= 0
 c2615: aux_cap_helper__b8,r13,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b8,r13,band_UNPAIRED__Z2 <= 0
 c2616: aux_cap_helper__b0,r6,band_UNPAIRED__condX2
        + aux_cap_helper__b0,r6,band_UNPAIRED__condX1 - X__b0,r6,band_UNPAIRED
         = 0
 c2617: 5.65753539715363 sv_function_helpervar_b1,r5__Z3
        - sv_function_helpervar_b1,r5__condX3 <= 0
 c2618: aux_cap_helper__b0,r10,band_UNPAIRED__Z2
        + aux_cap_helper__b0,r10,band_UNPAIRED__Z1  = 1
 c2619: aux_cap_helper__b9,r8,band_LOW_PAIRED__Z2
        - aux_cap_helper__b9,r8,band_LOW_PAIRED__condX2 <= 0
 c2620: - 2 aux_cap_helper__b8,r9,band_UNPAIRED__Z2
        + aux_cap_helper__b8,r9,band_UNPAIRED__condX2 <= 0
 c2621: - aux_cap_helper__b0,r12,band_UNPAIRED__Z1
        + aux_cap_helper__b0,r12,band_UNPAIRED__condX1 <= 0
 c2622: - sv_function_helpervar_b8,r11__condX1 <= 0
 c2623: - aux_cap_helper__b2,r0,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b2,r0,band_HIGH_PAIRED__condX1 <= 0
 c2624: aux_cap_helper__b5,r2,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b5,r2,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b5,r2,band_HIGH_PAIRED__Z3  = 1
 c2625: sv_function_helpervar_b1,r4__condX2
        - 6.26387748824355 sv_function_helpervar_b1,r4__Z2 <= 0
 c2626: - aux_Omega_b7,r11 + 2900.66054316859 aux_sv_b7,r11  = 0
 c2627: - X__b1,r4,band_LOW_PAIRED
        + aux_cap_helper__b1,r4,band_LOW_PAIRED__condX2
        + aux_cap_helper__b1,r4,band_LOW_PAIRED__condX1  = 0
 c2628: - X__b5,r1,band_LOW_PAIRED
        + aux_cap_helper__b5,r1,band_LOW_PAIRED__condX2
        + aux_cap_helper__b5,r1,band_LOW_PAIRED__condX1  = 0
 c2629: - aux_c_b7,r3 + sv_function_helpervar_b7,r3__condX3
        + sv_function_helpervar_b7,r3__condX1
        + sv_function_helpervar_b7,r3__condX2  = 0
 c2630: aux_cap_helper__b9,r9,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b9,r9,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b9,r9,band_HIGH_PAIRED__condX1
        - X__b9,r9,band_HIGH_PAIRED  = 0
 c2631: - sv_function_helpervar_b8,r1__condX1 <= 0
 c2632: - 2 aux_cap_helper__b2,r0,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b2,r0,band_HIGH_PAIRED__condX2 <= 0
 c2633: - 2 aux_cap_helper__b0,r12,band_UNPAIRED__Z2
        + aux_cap_helper__b0,r12,band_UNPAIRED__condX2 <= 0
 c2634: aux_cap_helper__b9,r10,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b9,r10,band_LOW_PAIRED__Z2 <= 0
 c2635: X__b8,r1,band_LOW_PAIRED + X__b8,r1,band_HIGH_PAIRED
        + X__b8,r1,band_UNPAIRED - GlobalBidder_W_ir__b8,r1 >= 0
 c2636: 0.159946187436519 sv_function_helpervar_b6,r4__Z2
        - sv_function_helpervar_b6,r4__condX2 <= 0
 c2637: 21.4536858652 sv_function_helpervar_b3,r13__condX3
        + 67.745769852 sv_function_helpervar_b3,r13__condX2
        + 1379.5885616567 sv_function_helpervar_b3,r13__condX1 - aux_sv_b3,r13
        + 209.824253080203 sv_function_helpervar_b3,r13__Z2
        + 474.115476480955 sv_function_helpervar_b3,r13__Z3  = 0
 c2638: aux_cap_helper__b7,r6,band_UNPAIRED__Z1
        + aux_cap_helper__b7,r6,band_UNPAIRED__Z2  = 1
 c2639: - 6.34442616466423 sv_function_helpervar_b1,r3__Z2
        + sv_function_helpervar_b1,r3__condX2 <= 0
 c2640: - 0.159946187462122 sv_function_helpervar_b4,r6__Z1
        + sv_function_helpervar_b4,r6__condX1 <= 0
 c2641: aux_cap_b5,r0,band_HIGH_PAIRED + aux_cap_b5,r0,band_UNPAIRED
        - aux_c_b5,r0 + aux_cap_b5,r0,band_LOW_PAIRED  = 0
 c2642: aux_cap_helper__b5,r9,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b5,r9,band_UNPAIRED__Z2 <= 0
 c2643: 0.159946187535603 sv_function_helpervar_b0,r6__Z2
        - sv_function_helpervar_b0,r6__condX2 <= 0
 c2644: - X__b6,r12,band_UNPAIRED
        + aux_cap_helper__b6,r12,band_UNPAIRED__condX2
        + aux_cap_helper__b6,r12,band_UNPAIRED__condX1  = 0
 c2645: aux_cap_helper__b1,r4,band_UNPAIRED__Z2
        + aux_cap_helper__b1,r4,band_UNPAIRED__Z1  = 1
 c2646: aux_cap_helper__b9,r12,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b9,r12,band_HIGH_PAIRED__Z3 <= 0
 c2647: aux_cap_helper__b1,r0,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b1,r0,band_UNPAIRED__Z2 <= 0
 c2648: aux_cap_helper__b2,r3,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b2,r3,band_HIGH_PAIRED__Z2 <= 0
 c2649: - aux_cap_helper__b8,r9,band_UNPAIRED__Z1
        + aux_cap_helper__b8,r9,band_UNPAIRED__condX1 <= 0
 c2650: sv_function_helpervar_b0,r4__condX3
        + sv_function_helpervar_b0,r4__condX2
        + sv_function_helpervar_b0,r4__condX1 - aux_c_b0,r4  = 0
 c2651: aux_cap_helper__b6,r13,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b6,r13,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b6,r13,band_HIGH_PAIRED__Z3  = 1
 c2652: sv_function_helpervar_b0,r3__condX3
        - 15.9946187432626 sv_function_helpervar_b0,r3__Z3 <= 0
 c2653: sv_function_helpervar_b2,r0__condX1
        - 0.159946187434522 sv_function_helpervar_b2,r0__Z1 <= 0
 c2654: 0.159946187459929 sv_function_helpervar_b7,r4__Z2
        - sv_function_helpervar_b7,r4__condX2 <= 0
 c2655: sv_function_helpervar_b5,r9__condX3
        + sv_function_helpervar_b5,r9__condX2
        + sv_function_helpervar_b5,r9__condX1 - aux_c_b5,r9  = 0
 c2656: sv_function_helpervar_b2,r1__condX2
        - 5.90199764671298 sv_function_helpervar_b2,r1__Z2 <= 0
 c2657: - 2 aux_cap_helper__b5,r0,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b5,r0,band_HIGH_PAIRED__condX2 <= 0
 c2658: aux_cap_helper__b3,r7,band_UNPAIRED__condX1
        + aux_cap_helper__b3,r7,band_UNPAIRED__condX2 - X__b3,r7,band_UNPAIRED
         = 0
 c2659: - aux_cap_helper__b9,r1,band_LOW_PAIRED__Z1
        + aux_cap_helper__b9,r1,band_LOW_PAIRED__condX1 <= 0
 c2660: aux_cap_helper__b9,r10,band_LOW_PAIRED__condX1
        - aux_cap_helper__b9,r10,band_LOW_PAIRED__Z1 <= 0
 c2661: aux_cap_helper__b6,r13,band_UNPAIRED__Z2
        - aux_cap_helper__b6,r13,band_UNPAIRED__condX2 <= 0
 c2662: sv_function_helpervar_b9,r6__condX2
        + sv_function_helpervar_b9,r6__condX3
        + sv_function_helpervar_b9,r6__condX1 - aux_c_b9,r6  = 0
 c2663: aux_cap_helper__b8,r11,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b8,r11,band_LOW_PAIRED__Z2 <= 0
 c2664: aux_cap_helper__b8,r2,band_LOW_PAIRED__Z2
        - aux_cap_helper__b8,r2,band_LOW_PAIRED__condX2 <= 0
 c2665: - sv_function_helpervar_b5,r4__condX3
        + 5.69598108125028 sv_function_helpervar_b5,r4__Z3 <= 0
 c2666: aux_cap_b6,r2,band_UNPAIRED - aux_c_b6,r2
        + aux_cap_b6,r2,band_LOW_PAIRED + aux_cap_b6,r2,band_HIGH_PAIRED  = 0
 c2667: - sv_function_helpervar_b5,r7__condX1 <= 0
 c2668: aux_cap_helper__b7,r12,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b7,r12,band_HIGH_PAIRED__Z1 <= 0
 c2669: sv_function_helpervar_b3,r9__condX2
        - 5.77656638381451 sv_function_helpervar_b3,r9__Z2 <= 0
 c2670: - 6.9785169243679 sv_function_helpervar_b9,r1__Z2
        + sv_function_helpervar_b9,r1__condX2 <= 0
 c2671: aux_Omega_b7,r10 + aux_Omega_b7,r11 + aux_Omega_b7,r12
        + aux_Omega_b7,r13 + aux_Omega_b7,r0 + aux_Omega_b7,r1
        + aux_Omega_b7,r2 + aux_Omega_b7,r3 + aux_Omega_b7,r4 + aux_Omega_b7,r5
        + aux_Omega_b7,r6 + aux_Omega_b7,r7 + aux_Omega_b7,r8 + aux_Omega_b7,r9
        - GlobalBidder_PSI__i7,k0 + 18497526.8940961 GlobalBidder_hat{W}__i7,k0
        <= 18497526.8940961
 c2672: aux_cap_helper__b8,r5,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b8,r5,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b8,r5,band_HIGH_PAIRED__Z3  = 1
 c2673: sv_function_helpervar_b6,r10__Z1 + sv_function_helpervar_b6,r10__Z3
        + sv_function_helpervar_b6,r10__Z2  = 1
 c2674: aux_cap_helper__b2,r3,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b2,r3,band_HIGH_PAIRED__Z1 <= 0
 c2675: - sv_function_helpervar_b2,r1__condX1 <= 0
 c2676: sv_function_helpervar_b6,r1__Z2 + sv_function_helpervar_b6,r1__Z3
        + sv_function_helpervar_b6,r1__Z1  = 1
 c2677: aux_cap_helper__b1,r0,band_UNPAIRED__condX1
        - aux_cap_helper__b1,r0,band_UNPAIRED__Z1 <= 0
 c2678: sv_function_helpervar_b3,r10__condX1
        - 0.159946187535581 sv_function_helpervar_b3,r10__Z1 <= 0
 c2679: - aux_Omega_b1,r4 + 1573.14548115063 aux_sv_b1,r4  = 0
 c2680: - aux_Omega_b9,r4 + 1853.54073373768 aux_sv_b9,r4  = 0
 c2681: - 1.4943850124 aux_cap_helper__b7,r9,band_LOW_PAIRED__Z2
        - aux_cap_b7,r9,band_LOW_PAIRED
        + 5.2303475434 aux_cap_helper__b7,r9,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b7,r9,band_LOW_PAIRED__condX1  = 0
 c2682: 76.9770902135393 aux_sv_b8,r6 - aux_Omega_b8,r6  = 0
 c2683: - sv_function_helpervar_b7,r6__condX3
        + 7.56076262079736 sv_function_helpervar_b7,r6__Z3 <= 0
 c2684: - aux_c_b2,r7 + sv_function_helpervar_b2,r7__condX3
        + sv_function_helpervar_b2,r7__condX2
        + sv_function_helpervar_b2,r7__condX1  = 0
 c2685: aux_cap_helper__b4,r10,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b4,r10,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b4,r10,band_HIGH_PAIRED__Z2  = 1
 c2686: sv_function_helpervar_b9,r3__condX2
        + sv_function_helpervar_b9,r3__condX1
        + sv_function_helpervar_b9,r3__condX3 - aux_c_b9,r3  = 0
 c2687: - aux_cap_helper__b0,r1,band_LOW_PAIRED__condX2
        + aux_cap_helper__b0,r1,band_LOW_PAIRED__Z2 <= 0
 c2688: - sv_function_helpervar_b4,r5__condX3
        + 6.37510953535038 sv_function_helpervar_b4,r5__Z3 <= 0
 c2689: - aux_cap_helper__b5,r0,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b5,r0,band_HIGH_PAIRED__condX1 <= 0
 c2690: - aux_cap_helper__b5,r1,band_UNPAIRED__condX1 <= 0
 c2691: aux_cap_helper__b5,r2,band_LOW_PAIRED__condX1
        - aux_cap_helper__b5,r2,band_LOW_PAIRED__Z1 <= 0
 c2692: - 15.9946187432626 sv_function_helpervar_b7,r0__Z3
        + sv_function_helpervar_b7,r0__condX3 <= 0
 c2693: - aux_cap_helper__b6,r2,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b6,r2,band_HIGH_PAIRED__Z3 <= 0
 c2694: - X__b0,r2,band_HIGH_PAIRED
        + aux_cap_helper__b0,r2,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b0,r2,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b0,r2,band_HIGH_PAIRED__condX1  = 0
 c2695: aux_cap_helper__b3,r12,band_UNPAIRED__condX2
        + aux_cap_helper__b3,r12,band_UNPAIRED__condX1
        - X__b3,r12,band_UNPAIRED  = 0
 c2696: - sv_function_helpervar_b0,r5__condX1 <= 0
 c2697: - sv_function_helpervar_b1,r11__condX1 <= 0
 c2698: - aux_cap_helper__b2,r9,band_LOW_PAIRED__Z1
        + aux_cap_helper__b2,r9,band_LOW_PAIRED__condX1 <= 0
 c2699: - aux_cap_helper__b0,r2,band_UNPAIRED__condX1 <= 0
 c2700: - aux_cap_helper__b5,r12,band_HIGH_PAIRED__condX1 <= 0
 c2701: - X__b1,r0,band_HIGH_PAIRED
        + aux_cap_helper__b1,r0,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b1,r0,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b1,r0,band_HIGH_PAIRED__condX1  = 0
 c2702: sv_function_helpervar_b4,r11__Z3 + sv_function_helpervar_b4,r11__Z1
        + sv_function_helpervar_b4,r11__Z2  = 1
 c2703: - X__b8,r5,band_UNPAIRED + aux_cap_helper__b8,r5,band_UNPAIRED__condX2
        + aux_cap_helper__b8,r5,band_UNPAIRED__condX1  = 0
 c2704: - 1.4943850124 aux_cap_helper__b2,r4,band_LOW_PAIRED__Z2
        + 3.735962531 aux_cap_helper__b2,r4,band_LOW_PAIRED__condX1
        + 5.2303475434 aux_cap_helper__b2,r4,band_LOW_PAIRED__condX2
        - aux_cap_b2,r4,band_LOW_PAIRED  = 0
 c2705: - 18497526.8940961 GlobalBidder_hat{W}__i8,k2 - GlobalBidder_PSI__i8,k2
        + aux_Omega_b8,r12 + aux_Omega_b8,r13 + aux_Omega_b8,r10
        + aux_Omega_b8,r11 + aux_Omega_b8,r5 + aux_Omega_b8,r6
        + aux_Omega_b8,r7 + aux_Omega_b8,r8 + aux_Omega_b8,r9 + aux_Omega_b8,r0
        + aux_Omega_b8,r1 + aux_Omega_b8,r2 + aux_Omega_b8,r3 + aux_Omega_b8,r4
        >= -18497526.8940961
 c2706: aux_cap_helper__b5,r2,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b5,r2,band_LOW_PAIRED__Z2 <= 0
 c2707: - aux_c_b0,r7 + sv_function_helpervar_b0,r7__condX1
        + sv_function_helpervar_b0,r7__condX2
        + sv_function_helpervar_b0,r7__condX3  = 0
 c2708: - 3 aux_cap_helper__b6,r5,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b6,r5,band_HIGH_PAIRED__condX3 <= 0
 c2709: - aux_c_b2,r4 + sv_function_helpervar_b2,r4__condX3
        + sv_function_helpervar_b2,r4__condX1
        + sv_function_helpervar_b2,r4__condX2  = 0
 c2710: - 2 aux_cap_helper__b2,r9,band_LOW_PAIRED__Z2
        + aux_cap_helper__b2,r9,band_LOW_PAIRED__condX2 <= 0
 c2711: - aux_cap_helper__b8,r6,band_HIGH_PAIRED__condX1 <= 0
 c2712: sv_function_helpervar_b7,r5__condX2
        - 6.42469180122383 sv_function_helpervar_b7,r5__Z2 <= 0
 c2713: aux_cap_helper__b9,r9,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b9,r9,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b9,r9,band_HIGH_PAIRED__Z1  = 1
 c2714: - aux_c_b7,r9 + sv_function_helpervar_b7,r9__condX1
        + sv_function_helpervar_b7,r9__condX2
        + sv_function_helpervar_b7,r9__condX3  = 0
 c2715: - aux_c_b7,r0 + sv_function_helpervar_b7,r0__condX1
        + sv_function_helpervar_b7,r0__condX2
        + sv_function_helpervar_b7,r0__condX3  = 0
 c2716: aux_cap_helper__b5,r11,band_UNPAIRED__condX1
        - aux_cap_helper__b5,r11,band_UNPAIRED__Z1 <= 0
 c2717: aux_cap_helper__b2,r0,band_LOW_PAIRED__Z2
        + aux_cap_helper__b2,r0,band_LOW_PAIRED__Z1  = 1
 c2718: sv_function_helpervar_b8,r11__condX1
        - 0.159946187501739 sv_function_helpervar_b8,r11__Z1 <= 0
 c2719: - aux_cap_helper__b3,r7,band_UNPAIRED__condX2
        + aux_cap_helper__b3,r7,band_UNPAIRED__Z2 <= 0
 c2720: aux_cap_helper__b5,r10,band_UNPAIRED__Z2
        + aux_cap_helper__b5,r10,band_UNPAIRED__Z1  = 1
 c2721: aux_cap_helper__b2,r10,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b2,r10,band_HIGH_PAIRED__Z3 <= 0
 c2722: aux_cap_helper__b3,r6,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b3,r6,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b3,r6,band_HIGH_PAIRED__Z3  = 1
 c2723: - 2 aux_cap_helper__b2,r13,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b2,r13,band_HIGH_PAIRED__condX2 <= 0
 c2724: aux_cap_helper__b7,r12,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b7,r12,band_HIGH_PAIRED__Z2 <= 0
 c2725: - aux_cap_helper__b6,r2,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b6,r2,band_HIGH_PAIRED__Z2 <= 0
 c2726: - sv_function_helpervar_b0,r2__condX2
        + 0.159946187506047 sv_function_helpervar_b0,r2__Z2 <= 0
 c2727: aux_cap_helper__b5,r3,band_LOW_PAIRED__Z2
        + aux_cap_helper__b5,r3,band_LOW_PAIRED__Z1  = 1
 c2728: - 3 aux_cap_helper__b6,r8,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b6,r8,band_HIGH_PAIRED__condX3 <= 0
 c2729: sv_function_helpervar_b8,r4__Z3 + sv_function_helpervar_b8,r4__Z2
        + sv_function_helpervar_b8,r4__Z1  = 1
 c2730: 5.2303475434 aux_cap_helper__b9,r12,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b9,r12,band_LOW_PAIRED__condX1
        - 1.4943850124 aux_cap_helper__b9,r12,band_LOW_PAIRED__Z2
        - aux_cap_b9,r12,band_LOW_PAIRED  = 0
 c2731: aux_cap_helper__b6,r8,band_LOW_PAIRED__condX1
        - aux_cap_helper__b6,r8,band_LOW_PAIRED__Z1 <= 0
 c2732: - X__b9,r10,band_LOW_PAIRED
        + aux_cap_helper__b9,r10,band_LOW_PAIRED__condX1
        + aux_cap_helper__b9,r10,band_LOW_PAIRED__condX2  = 0
 c2733: - 6.04691642650674 sv_function_helpervar_b3,r0__Z2
        + sv_function_helpervar_b3,r0__condX2 <= 0
 c2734: - aux_Omega_b4,r8 + 178.088355903143 aux_sv_b4,r8  = 0
 c2735: 6.6833780439137 sv_function_helpervar_b8,r9__Z3
        - sv_function_helpervar_b8,r9__condX3 <= 0
 c2736: aux_cap_helper__b3,r4,band_LOW_PAIRED__condX2
        + aux_cap_helper__b3,r4,band_LOW_PAIRED__condX1
        - X__b3,r4,band_LOW_PAIRED  = 0
 c2737: sv_function_helpervar_b3,r9__condX1
        - 0.159946187458878 sv_function_helpervar_b3,r9__Z1 <= 0
 c2738: aux_cap_helper__b1,r5,band_LOW_PAIRED__Z1
        + aux_cap_helper__b1,r5,band_LOW_PAIRED__Z2  = 1
 c2739: - 1.4943850124 aux_cap_helper__b6,r6,band_LOW_PAIRED__Z2
        + 5.2303475434 aux_cap_helper__b6,r6,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b6,r6,band_LOW_PAIRED__condX1
        - aux_cap_b6,r6,band_LOW_PAIRED  = 0
 c2740: aux_cap_b8,r11,band_LOW_PAIRED + aux_cap_b8,r11,band_UNPAIRED
        + aux_cap_b8,r11,band_HIGH_PAIRED - aux_c_b8,r11  = 0
 c2741: - aux_cap_helper__b0,r8,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b0,r8,band_HIGH_PAIRED__Z2 <= 0
 c2742: aux_cap_helper__b7,r1,band_LOW_PAIRED__condX1
        + aux_cap_helper__b7,r1,band_LOW_PAIRED__condX2
        - X__b7,r1,band_LOW_PAIRED  = 0
 c2743: - aux_cap_helper__b9,r8,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b9,r8,band_HIGH_PAIRED__condX1 <= 0
 c2744: aux_cap_helper__b1,r3,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b1,r3,band_LOW_PAIRED__Z2 <= 0
 c2745: - aux_cap_helper__b5,r6,band_LOW_PAIRED__condX2
        + aux_cap_helper__b5,r6,band_LOW_PAIRED__Z2 <= 0
 c2746: - aux_cap_helper__b0,r8,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b0,r8,band_HIGH_PAIRED__Z3 <= 0
 c2747: - 0.3214355447 aux_cap_helper__b8,r9,band_UNPAIRED__Z2
        - aux_cap_b8,r9,band_UNPAIRED
        + 1.1250244065 aux_cap_helper__b8,r9,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b8,r9,band_UNPAIRED__condX1  = 0
 c2748: 6.37162279776951 sv_function_helpervar_b0,r1__Z3
        - sv_function_helpervar_b0,r1__condX3 <= 0
 c2749: - 0.159946187451652 sv_function_helpervar_b5,r12__Z1
        + sv_function_helpervar_b5,r12__condX1 <= 0
 c2750: aux_cap_helper__b8,r6,band_LOW_PAIRED__Z2
        + aux_cap_helper__b8,r6,band_LOW_PAIRED__Z1  = 1
 c2751: - 3 aux_cap_helper__b9,r5,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b9,r5,band_HIGH_PAIRED__condX3 <= 0
 c2752: - X__b3,r9,band_HIGH_PAIRED
        + aux_cap_helper__b3,r9,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b3,r9,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b3,r9,band_HIGH_PAIRED__condX1  = 0
 c2753: - aux_cap_helper__b3,r2,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b3,r2,band_HIGH_PAIRED__Z3 <= 0
 c2754: - 2 aux_cap_helper__b9,r8,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b9,r8,band_HIGH_PAIRED__condX2 <= 0
 c2755: aux_cap_helper__b0,r5,band_UNPAIRED__Z2
        + aux_cap_helper__b0,r5,band_UNPAIRED__Z1  = 1
 c2756: aux_cap_helper__b2,r12,band_LOW_PAIRED__condX2
        + aux_cap_helper__b2,r12,band_LOW_PAIRED__condX1
        - X__b2,r12,band_LOW_PAIRED  = 0
 c2757: 0.0102040816326531 X__b7,r2,band_HIGH_PAIRED - GlobalBidder_W_ir__b7,r2
        + 0.0102040816326531 X__b7,r2,band_LOW_PAIRED
        + 0.0102040816326531 X__b7,r2,band_UNPAIRED <= 0
 c2758: aux_cap_helper__b4,r0,band_LOW_PAIRED__Z2
        - aux_cap_helper__b4,r0,band_LOW_PAIRED__condX2 <= 0
 c2759: - aux_cap_helper__b8,r9,band_HIGH_PAIRED__condX1 <= 0
 c2760: - aux_cap_helper__b3,r2,band_UNPAIRED__condX1 <= 0
 c2761: - aux_cap_b6,r6,band_UNPAIRED
        + 0.8035888618 aux_cap_helper__b6,r6,band_UNPAIRED__condX1
        + 1.1250244065 aux_cap_helper__b6,r6,band_UNPAIRED__condX2
        - 0.3214355447 aux_cap_helper__b6,r6,band_UNPAIRED__Z2  = 0
 c2762: aux_cap_helper__b3,r11,band_LOW_PAIRED__Z2
        + aux_cap_helper__b3,r11,band_LOW_PAIRED__Z1  = 1
 c2763: - aux_cap_helper__b2,r5,band_LOW_PAIRED__condX1 <= 0
 c2764: - aux_cap_helper__b5,r9,band_HIGH_PAIRED__condX1 <= 0
 c2765: - sv_function_helpervar_b5,r6__condX1 <= 0
 c2766: - sv_function_helpervar_b4,r4__condX2
        + 0.159946187455554 sv_function_helpervar_b4,r4__Z2 <= 0
 c2767: aux_cap_helper__b6,r7,band_UNPAIRED__Z2
        - aux_cap_helper__b6,r7,band_UNPAIRED__condX2 <= 0
 c2768: - aux_sv_b7,r5 + 30.9252268385 sv_function_helpervar_b7,r5__condX3
        + 80.4844801001 sv_function_helpervar_b7,r5__condX2
        + 1850.3233257314 sv_function_helpervar_b7,r5__condX1
        + 283.078975768328 sv_function_helpervar_b7,r5__Z2
        + 601.481903872901 sv_function_helpervar_b7,r5__Z3  = 0
 c2769: aux_cap_helper__b5,r11,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b5,r11,band_UNPAIRED__Z2 <= 0
 c2770: - aux_c_b7,r6 + sv_function_helpervar_b7,r6__condX3
        + sv_function_helpervar_b7,r6__condX2
        + sv_function_helpervar_b7,r6__condX1  = 0
 c2771: sv_function_helpervar_b2,r7__Z1 + sv_function_helpervar_b2,r7__Z2
        + sv_function_helpervar_b2,r7__Z3  = 1
 c2772: - aux_cap_helper__b3,r5,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b3,r5,band_HIGH_PAIRED__Z3 <= 0
 c2773: aux_cap_helper__b4,r12,band_UNPAIRED__Z1
        + aux_cap_helper__b4,r12,band_UNPAIRED__Z2  = 1
 c2774: - aux_cap_helper__b7,r1,band_LOW_PAIRED__condX1 <= 0
 c2775: 5.2303475434 aux_cap_helper__b5,r3,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b5,r3,band_LOW_PAIRED__condX1
        - aux_cap_b5,r3,band_LOW_PAIRED
        - 1.4943850124 aux_cap_helper__b5,r3,band_LOW_PAIRED__Z2  = 0
 c2776: aux_cap_helper__b4,r0,band_UNPAIRED__Z2
        + aux_cap_helper__b4,r0,band_UNPAIRED__Z1  = 1
 c2777: - 3 aux_cap_helper__b9,r2,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b9,r2,band_HIGH_PAIRED__condX3 <= 0
 c2778: - aux_c_b8,r3 + aux_cap_b8,r3,band_UNPAIRED
        + aux_cap_b8,r3,band_HIGH_PAIRED + aux_cap_b8,r3,band_LOW_PAIRED  = 0
 c2779: 0.8035888618 aux_cap_helper__b7,r2,band_UNPAIRED__condX1
        + 1.1250244065 aux_cap_helper__b7,r2,band_UNPAIRED__condX2
        - 0.3214355447 aux_cap_helper__b7,r2,band_UNPAIRED__Z2
        - aux_cap_b7,r2,band_UNPAIRED  = 0
 c2780: aux_cap_helper__b0,r3,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b0,r3,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b0,r3,band_HIGH_PAIRED__Z2  = 1
 c2781: 1.1250244065 aux_cap_helper__b4,r0,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b4,r0,band_UNPAIRED__condX1
        - aux_cap_b4,r0,band_UNPAIRED
        - 0.3214355447 aux_cap_helper__b4,r0,band_UNPAIRED__Z2  = 0
 c2782: 2 aux_cap_helper__b3,r8,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b3,r8,band_HIGH_PAIRED__condX3 <= 0
 c2783: - X__b2,r6,band_LOW_PAIRED
        + aux_cap_helper__b2,r6,band_LOW_PAIRED__condX2
        + aux_cap_helper__b2,r6,band_LOW_PAIRED__condX1  = 0
 c2784: aux_cap_helper__b9,r7,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b9,r7,band_HIGH_PAIRED__Z3 <= 0
 c2785: - aux_cap_helper__b3,r12,band_LOW_PAIRED__condX1 <= 0
 c2786: - aux_cap_helper__b9,r0,band_UNPAIRED__condX1 <= 0
 c2787: sv_function_helpervar_b3,r12__condX3
        + sv_function_helpervar_b3,r12__condX2
        + sv_function_helpervar_b3,r12__condX1 - aux_c_b3,r12  = 0
 c2788: sv_function_helpervar_b9,r9__condX1
        + sv_function_helpervar_b9,r9__condX2
        + sv_function_helpervar_b9,r9__condX3 - aux_c_b9,r9  = 0
 c2789: aux_cap_helper__b6,r4,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b6,r4,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b6,r4,band_HIGH_PAIRED__Z1  = 1
 c2790: aux_cap_helper__b3,r9,band_UNPAIRED__condX1
        - aux_cap_helper__b3,r9,band_UNPAIRED__Z1 <= 0
 c2791: - aux_c_b5,r6 + sv_function_helpervar_b5,r6__condX1
        + sv_function_helpervar_b5,r6__condX2
        + sv_function_helpervar_b5,r6__condX3  = 0
 c2792: - X__b8,r3,band_HIGH_PAIRED
        + aux_cap_helper__b8,r3,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b8,r3,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b8,r3,band_HIGH_PAIRED__condX1  = 0
 c2793: - aux_cap_helper__b0,r11,band_UNPAIRED__Z1
        + aux_cap_helper__b0,r11,band_UNPAIRED__condX1 <= 0
 c2794: - sv_function_helpervar_b9,r13__condX2
        + 0.159946187446348 sv_function_helpervar_b9,r13__Z2 <= 0
 c2795: sv_function_helpervar_b0,r2__condX3
        - 15.9946187432626 sv_function_helpervar_b0,r2__Z3 <= 0
 c2796: - sv_function_helpervar_b3,r13__condX2
        + 0.159946187447925 sv_function_helpervar_b3,r13__Z2 <= 0
 c2797: - aux_cap_b7,r2,band_LOW_PAIRED
        + 5.2303475434 aux_cap_helper__b7,r2,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b7,r2,band_LOW_PAIRED__condX1
        - 1.4943850124 aux_cap_helper__b7,r2,band_LOW_PAIRED__Z2  = 0
 c2798: - aux_cap_helper__b5,r1,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b5,r1,band_HIGH_PAIRED__condX1 <= 0
 c2799: sv_function_helpervar_b5,r13__Z3 + sv_function_helpervar_b5,r13__Z2
        + sv_function_helpervar_b5,r13__Z1  = 1
 c2800: aux_cap_helper__b0,r3,band_LOW_PAIRED__condX1
        - aux_cap_helper__b0,r3,band_LOW_PAIRED__Z1 <= 0
 c2801: aux_cap_helper__b3,r1,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b3,r1,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b3,r1,band_HIGH_PAIRED__condX1
        - X__b3,r1,band_HIGH_PAIRED  = 0
 c2802: - sv_function_helpervar_b8,r10__condX1 <= 0
 c2803: aux_cap_helper__b4,r9,band_LOW_PAIRED__Z1
        + aux_cap_helper__b4,r9,band_LOW_PAIRED__Z2  = 1
 c2804: - X__b8,r7,band_UNPAIRED + aux_cap_helper__b8,r7,band_UNPAIRED__condX1
        + aux_cap_helper__b8,r7,band_UNPAIRED__condX2  = 0
 c2805: - sv_function_helpervar_b2,r2__condX2
        + 0.159946187457741 sv_function_helpervar_b2,r2__Z2 <= 0
 c2806: sv_function_helpervar_b6,r11__condX1
        - 0.159946187466827 sv_function_helpervar_b6,r11__Z1 <= 0
 c2807: - 2 aux_cap_helper__b6,r2,band_LOW_PAIRED__Z2
        + aux_cap_helper__b6,r2,band_LOW_PAIRED__condX2 <= 0
 c2808: - X__b5,r6,band_UNPAIRED + aux_cap_helper__b5,r6,band_UNPAIRED__condX1
        + aux_cap_helper__b5,r6,band_UNPAIRED__condX2  = 0
 c2809: - aux_cap_helper__b5,r7,band_HIGH_PAIRED__condX1 <= 0
 c2810: 778.283786242301 aux_sv_b8,r5 - aux_Omega_b8,r5  = 0
 c2811: - aux_cap_helper__b0,r5,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b0,r5,band_HIGH_PAIRED__Z2 <= 0
 c2812: - aux_cap_helper__b1,r2,band_LOW_PAIRED__condX1 <= 0
 c2813: aux_cap_helper__b3,r10,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b3,r10,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b3,r10,band_HIGH_PAIRED__Z3  = 1
 c2814: - aux_cap_helper__b8,r11,band_UNPAIRED__condX2
        + aux_cap_helper__b8,r11,band_UNPAIRED__Z2 <= 0
 c2815: 528.111003731967 sv_function_helpervar_b9,r12__Z3
        + 264.046997361731 sv_function_helpervar_b9,r12__Z2 - aux_sv_b9,r12
        + 68.2456271259 sv_function_helpervar_b9,r12__condX2
        + 1719.0945876568 sv_function_helpervar_b9,r12__condX1
        + 30.6521272933 sv_function_helpervar_b9,r12__condX3  = 0
 c2816: - aux_cap_helper__b5,r3,band_LOW_PAIRED__condX2
        + aux_cap_helper__b5,r3,band_LOW_PAIRED__Z2 <= 0
 c2817: 0.927217345653617 aux_cap_helper__b0,r12,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b0,r12,band_HIGH_PAIRED__Z2
        - aux_cap_b0,r12,band_HIGH_PAIRED
        + 2.1635071398 aux_cap_helper__b0,r12,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b0,r12,band_HIGH_PAIRED__condX1
        + 1.3908260184 aux_cap_helper__b0,r12,band_HIGH_PAIRED__condX3  = 0
 c2818: sv_function_helpervar_b0,r1__condX1
        + sv_function_helpervar_b0,r1__condX3
        + sv_function_helpervar_b0,r1__condX2 - aux_c_b0,r1  = 0
 c2819: aux_cap_helper__b2,r10,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b2,r10,band_HIGH_PAIRED__Z2 <= 0
 c2820: aux_cap_helper__b1,r6,band_LOW_PAIRED__Z2
        + aux_cap_helper__b1,r6,band_LOW_PAIRED__Z1  = 1
 c2821: aux_cap_helper__b2,r3,band_UNPAIRED__Z1
        + aux_cap_helper__b2,r3,band_UNPAIRED__Z2  = 1
 c2822: 2 aux_cap_helper__b4,r13,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b4,r13,band_HIGH_PAIRED__condX3 <= 0
 c2823: aux_cap_helper__b7,r8,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b7,r8,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b7,r8,band_HIGH_PAIRED__condX3
        - X__b7,r8,band_HIGH_PAIRED  = 0
 c2824: - aux_cap_helper__b6,r4,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b6,r4,band_HIGH_PAIRED__Z3 <= 0
 c2825: aux_cap_helper__b9,r13,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b9,r13,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b9,r13,band_HIGH_PAIRED__condX3
        - X__b9,r13,band_HIGH_PAIRED  = 0
 c2826: aux_cap_helper__b3,r0,band_UNPAIRED__condX1
        - aux_cap_helper__b3,r0,band_UNPAIRED__Z1 <= 0
 c2827: - aux_cap_helper__b7,r7,band_UNPAIRED__condX2
        + aux_cap_helper__b7,r7,band_UNPAIRED__Z2 <= 0
 c2828: aux_cap_helper__b9,r11,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b9,r11,band_HIGH_PAIRED__Z3 <= 0
 c2829: - aux_cap_helper__b9,r9,band_UNPAIRED__Z1
        + aux_cap_helper__b9,r9,band_UNPAIRED__condX1 <= 0
 c2830: - sv_function_helpervar_b1,r4__condX1 <= 0
 c2831: - X__b8,r12,band_HIGH_PAIRED
        + aux_cap_helper__b8,r12,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b8,r12,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b8,r12,band_HIGH_PAIRED__condX2  = 0
 c2832: aux_cap_helper__b7,r3,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b7,r3,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b7,r3,band_HIGH_PAIRED__Z3  = 1
 c2833: aux_cap_helper__b9,r1,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b9,r1,band_HIGH_PAIRED__Z3 <= 0
 c2834: aux_cap_b3,r7,band_UNPAIRED + aux_cap_b3,r7,band_LOW_PAIRED
        + aux_cap_b3,r7,band_HIGH_PAIRED - aux_c_b3,r7  = 0
 c2835: - aux_cap_helper__b6,r2,band_LOW_PAIRED__Z1
        + aux_cap_helper__b6,r2,band_LOW_PAIRED__condX1 <= 0
 c2836: - aux_cap_helper__b1,r1,band_HIGH_PAIRED__condX1 <= 0
 c2837: aux_cap_helper__b3,r0,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b3,r0,band_UNPAIRED__Z2 <= 0
 c2838: - aux_cap_helper__b1,r7,band_LOW_PAIRED__condX2
        + aux_cap_helper__b1,r7,band_LOW_PAIRED__Z2 <= 0
 c2839: - X__b2,r11,band_UNPAIRED
        + aux_cap_helper__b2,r11,band_UNPAIRED__condX2
        + aux_cap_helper__b2,r11,band_UNPAIRED__condX1  = 0
 c2840: - aux_cap_b3,r9,band_LOW_PAIRED
        - 1.4943850124 aux_cap_helper__b3,r9,band_LOW_PAIRED__Z2
        + 5.2303475434 aux_cap_helper__b3,r9,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b3,r9,band_LOW_PAIRED__condX1  = 0
 c2841: aux_cap_helper__b3,r9,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b3,r9,band_UNPAIRED__Z2 <= 0
 c2842: sv_function_helpervar_b3,r3__Z2 + sv_function_helpervar_b3,r3__Z1
        + sv_function_helpervar_b3,r3__Z3  = 1
 c2843: aux_cap_helper__b9,r4,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b9,r4,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b9,r4,band_HIGH_PAIRED__Z3  = 1
 c2844: aux_cap_helper__b0,r3,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b0,r3,band_LOW_PAIRED__Z2 <= 0
 c2845: - sv_function_helpervar_b6,r9__condX3
        + 6.20098161125853 sv_function_helpervar_b6,r9__Z3 <= 0
 c2846: aux_cap_helper__b8,r5,band_UNPAIRED__Z2
        + aux_cap_helper__b8,r5,band_UNPAIRED__Z1  = 1
 c2847: aux_cap_helper__b2,r10,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b2,r10,band_HIGH_PAIRED__Z1 <= 0
 c2848: - aux_cap_helper__b6,r4,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b6,r4,band_HIGH_PAIRED__Z2 <= 0
 c2849: aux_cap_helper__b4,r13,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b4,r13,band_HIGH_PAIRED__condX2 <= 0
 c2850: 0.159946187433943 sv_function_helpervar_b1,r0__Z2
        - sv_function_helpervar_b1,r0__condX2 <= 0
 c2851: sv_function_helpervar_b6,r2__condX1
        - 0.159946187456351 sv_function_helpervar_b6,r2__Z1 <= 0
 c2852: aux_cap_helper__b9,r2,band_UNPAIRED__Z1
        + aux_cap_helper__b9,r2,band_UNPAIRED__Z2  = 1
 c2853: 0.0102040816326531 X__b8,r4,band_HIGH_PAIRED
        + 0.0102040816326531 X__b8,r4,band_UNPAIRED
        + 0.0102040816326531 X__b8,r4,band_LOW_PAIRED
        - GlobalBidder_W_ir__b8,r4 <= 0
 c2854: - sv_function_helpervar_b3,r12__condX1 <= 0
 c2855: sv_function_helpervar_b7,r8__Z2 + sv_function_helpervar_b7,r8__Z3
        + sv_function_helpervar_b7,r8__Z1  = 1
 c2856: 284.949398601445 sv_function_helpervar_b7,r1__Z2
        + 539.543977326984 sv_function_helpervar_b7,r1__Z3
        + 68.7904043481 sv_function_helpervar_b7,r1__condX2
        + 1850.3233258617 sv_function_helpervar_b7,r1__condX1
        + 34.7976496539 sv_function_helpervar_b7,r1__condX3 - aux_sv_b7,r1  = 0
 c2857: 0.0102040816326531 X__b7,r6,band_LOW_PAIRED
        + 0.0102040816326531 X__b7,r6,band_UNPAIRED - GlobalBidder_W_ir__b7,r6
        + 0.0102040816326531 X__b7,r6,band_HIGH_PAIRED <= 0
 c2858: 0.0102040816326531 X__b9,r11,band_UNPAIRED - GlobalBidder_W_ir__b9,r11
        + 0.0102040816326531 X__b9,r11,band_HIGH_PAIRED
        + 0.0102040816326531 X__b9,r11,band_LOW_PAIRED <= 0
 c2859: - 2 aux_cap_helper__b5,r5,band_LOW_PAIRED__Z2
        + aux_cap_helper__b5,r5,band_LOW_PAIRED__condX2 <= 0
 c2860: - 5.97379865648078 sv_function_helpervar_b4,r0__Z2
        + sv_function_helpervar_b4,r0__condX2 <= 0
 c2861: X__b7,r6,band_LOW_PAIRED + X__b7,r6,band_UNPAIRED
        - GlobalBidder_W_ir__b7,r6 + X__b7,r6,band_HIGH_PAIRED >= 0
 c2862: 1.5453622427 aux_cap_helper__b0,r13,band_HIGH_PAIRED__condX1
        + 2.1635071398 aux_cap_helper__b0,r13,band_HIGH_PAIRED__condX2
        + 1.3908260184 aux_cap_helper__b0,r13,band_HIGH_PAIRED__condX3
        - 0.6181448971 aux_cap_helper__b0,r13,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b0,r13,band_HIGH_PAIRED__Z3
        - aux_cap_b0,r13,band_HIGH_PAIRED  = 0
 c2863: - X__b2,r6,band_HIGH_PAIRED
        + aux_cap_helper__b2,r6,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b2,r6,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b2,r6,band_HIGH_PAIRED__condX1  = 0
 c2864: - aux_cap_helper__b5,r8,band_HIGH_PAIRED__condX1 <= 0
 c2865: aux_cap_helper__b5,r11,band_UNPAIRED__Z2
        + aux_cap_helper__b5,r11,band_UNPAIRED__Z1  = 1
 c2866: 1850.3233248348 sv_function_helpervar_b7,r2__condX1
        + 34.0560628488 sv_function_helpervar_b7,r2__condX3
        + 70.5735777335 sv_function_helpervar_b7,r2__condX2
        + 284.66418681064 sv_function_helpervar_b7,r2__Z2
        + 551.405375539785 sv_function_helpervar_b7,r2__Z3 - aux_sv_b7,r2  = 0
 c2867: 0.159946187437218 sv_function_helpervar_b1,r8__Z2
        - sv_function_helpervar_b1,r8__condX2 <= 0
 c2868: - aux_cap_helper__b1,r8,band_UNPAIRED__condX2
        + aux_cap_helper__b1,r8,band_UNPAIRED__Z2 <= 0
 c2869: 711.146944848197 aux_sv_b1,r12 - aux_Omega_b1,r12  = 0
 c2870: aux_cap_helper__b3,r7,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b3,r7,band_HIGH_PAIRED__Z3 <= 0
 c2871: - sv_function_helpervar_b4,r13__condX2
        + 0.159946187484425 sv_function_helpervar_b4,r13__Z2 <= 0
 c2872: - X__b6,r10,band_HIGH_PAIRED
        + aux_cap_helper__b6,r10,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b6,r10,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b6,r10,band_HIGH_PAIRED__condX2  = 0
 c2873: - aux_cap_helper__b1,r0,band_LOW_PAIRED__Z1
        + aux_cap_helper__b1,r0,band_LOW_PAIRED__condX1 <= 0
 c2874: aux_cap_helper__b0,r6,band_UNPAIRED__Z1
        + aux_cap_helper__b0,r6,band_UNPAIRED__Z2  = 1
 c2875: - 15.9946187432626 sv_function_helpervar_b3,r11__Z3
        + sv_function_helpervar_b3,r11__condX3 <= 0
 c2876: 0.927217345653617 aux_cap_helper__b5,r13,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b5,r13,band_HIGH_PAIRED__Z2
        + 1.3908260184 aux_cap_helper__b5,r13,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b5,r13,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b5,r13,band_HIGH_PAIRED__condX1
        - aux_cap_b5,r13,band_HIGH_PAIRED  = 0
 c2877: - 3 aux_cap_helper__b2,r11,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b2,r11,band_HIGH_PAIRED__condX3 <= 0
 c2878: aux_cap_helper__b9,r1,band_LOW_PAIRED__Z1
        + aux_cap_helper__b9,r1,band_LOW_PAIRED__Z2  = 1
 c2879: sv_function_helpervar_b0,r4__Z1 + sv_function_helpervar_b0,r4__Z2
        + sv_function_helpervar_b0,r4__Z3  = 1
 c2880: aux_cap_helper__b4,r13,band_UNPAIRED__Z2
        + aux_cap_helper__b4,r13,band_UNPAIRED__Z1  = 1
 c2881: - 3 aux_cap_helper__b9,r0,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b9,r0,band_HIGH_PAIRED__condX3 <= 0
 c2882: 18497526.8940961 GlobalBidder_hat{W}__i8,k0 - GlobalBidder_PSI__i8,k0
        + aux_Omega_b8,r12 + aux_Omega_b8,r13 + aux_Omega_b8,r10
        + aux_Omega_b8,r11 + aux_Omega_b8,r5 + aux_Omega_b8,r6
        + aux_Omega_b8,r7 + aux_Omega_b8,r8 + aux_Omega_b8,r9 + aux_Omega_b8,r0
        + aux_Omega_b8,r1 + aux_Omega_b8,r2 + aux_Omega_b8,r3 + aux_Omega_b8,r4
        <= 18497526.8940961
 c2883: 5.2303475434 aux_cap_helper__b8,r12,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b8,r12,band_LOW_PAIRED__condX1
        - 1.4943850124 aux_cap_helper__b8,r12,band_LOW_PAIRED__Z2
        - aux_cap_b8,r12,band_LOW_PAIRED  = 0
 c2884: sv_function_helpervar_b1,r10__Z3 + sv_function_helpervar_b1,r10__Z2
        + sv_function_helpervar_b1,r10__Z1  = 1
 c2885: aux_cap_helper__b4,r3,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b4,r3,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b4,r3,band_HIGH_PAIRED__Z1  = 1
 c2886: - aux_cap_helper__b8,r13,band_LOW_PAIRED__condX1 <= 0
 c2887: 51.1609644922 sv_function_helpervar_b6,r6__condX2
        + 1059.4718671982 sv_function_helpervar_b6,r6__condX1
        + 16.6273343661 sv_function_helpervar_b6,r6__condX3
        + 161.275484693472 sv_function_helpervar_b6,r6__Z2
        + 361.676147996181 sv_function_helpervar_b6,r6__Z3 - aux_sv_b6,r6  = 0
 c2888: 0.159946187435134 sv_function_helpervar_b7,r10__Z2
        - sv_function_helpervar_b7,r10__condX2 <= 0
 c2889: - aux_cap_helper__b4,r6,band_UNPAIRED__condX2
        + aux_cap_helper__b4,r6,band_UNPAIRED__Z2 <= 0
 c2890: aux_cap_helper__b6,r3,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b6,r3,band_HIGH_PAIRED__condX2 <= 0
 c2891: - aux_cap_helper__b5,r2,band_UNPAIRED__condX1 <= 0
 c2892: - 2 aux_cap_helper__b9,r9,band_UNPAIRED__Z2
        + aux_cap_helper__b9,r9,band_UNPAIRED__condX2 <= 0
 c2893: - X__b8,r6,band_HIGH_PAIRED
        + aux_cap_helper__b8,r6,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b8,r6,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b8,r6,band_HIGH_PAIRED__condX3  = 0
 c2894: - sv_function_helpervar_b1,r12__condX1 <= 0
 c2895: aux_cap_b3,r10,band_LOW_PAIRED + aux_cap_b3,r10,band_UNPAIRED
        - aux_c_b3,r10 + aux_cap_b3,r10,band_HIGH_PAIRED  = 0
 c2896: aux_cap_helper__b0,r11,band_LOW_PAIRED__condX2
        + aux_cap_helper__b0,r11,band_LOW_PAIRED__condX1
        - X__b0,r11,band_LOW_PAIRED  = 0
 c2897: sv_function_helpervar_b8,r0__condX1
        - 0.159946187434592 sv_function_helpervar_b8,r0__Z1 <= 0
 c2898: aux_cap_helper__b5,r11,band_UNPAIRED__condX1
        + aux_cap_helper__b5,r11,band_UNPAIRED__condX2
        - X__b5,r11,band_UNPAIRED  = 0
 c2899: 0.00232 aux_Omega_b5,r1 + 0.00552 aux_Omega_b5,r0
        + 0.07433 aux_Omega_b5,r9 + 0.03125 aux_Omega_b5,r8
        + 0.01314 aux_Omega_b5,r7 + 0.01314 aux_Omega_b5,r6
        + 0.01314 aux_Omega_b5,r5 + 0.00552 aux_Omega_b5,r4
        + 0.00552 aux_Omega_b5,r3 + 0.00232 aux_Omega_b5,r2 - v__5
        + 0.42045 aux_Omega_b5,r13 + 0.17678 aux_Omega_b5,r10
        + 0.42045 aux_Omega_b5,r11 + aux_Omega_b5,r12  = 0
 c2900: 5.81605834545243 sv_function_helpervar_b6,r8__Z3
        - sv_function_helpervar_b6,r8__condX3 <= 0
 c2901: - 2 aux_cap_helper__b9,r1,band_LOW_PAIRED__Z2
        + aux_cap_helper__b9,r1,band_LOW_PAIRED__condX2 <= 0
 c2902: sv_function_helpervar_b3,r9__condX3
        + sv_function_helpervar_b3,r9__condX1
        + sv_function_helpervar_b3,r9__condX2 - aux_c_b3,r9  = 0
 c2903: 2523.07305925161 aux_sv_b0,r7 - aux_Omega_b0,r7  = 0
 c2904: - X__b0,r12,band_UNPAIRED
        + aux_cap_helper__b0,r12,band_UNPAIRED__condX1
        + aux_cap_helper__b0,r12,band_UNPAIRED__condX2  = 0
 c2905: - aux_cap_helper__b2,r13,band_LOW_PAIRED__condX1 <= 0
 c2906: - sv_function_helpervar_b7,r3__condX1 <= 0
 c2907: - aux_cap_helper__b2,r3,band_UNPAIRED__condX1 <= 0
 c2908: - aux_cap_b9,r8,band_UNPAIRED
        + 1.1250244065 aux_cap_helper__b9,r8,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b9,r8,band_UNPAIRED__condX1
        - 0.3214355447 aux_cap_helper__b9,r8,band_UNPAIRED__Z2  = 0
 c2909: - aux_cap_helper__b1,r7,band_UNPAIRED__condX2
        + aux_cap_helper__b1,r7,band_UNPAIRED__Z2 <= 0
 c2910: 0.8035888618 aux_cap_helper__b1,r11,band_UNPAIRED__condX1
        + 1.1250244065 aux_cap_helper__b1,r11,band_UNPAIRED__condX2
        - 0.3214355447 aux_cap_helper__b1,r11,band_UNPAIRED__Z2
        - aux_cap_b1,r11,band_UNPAIRED  = 0
 c2911: aux_cap_helper__b3,r3,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b3,r3,band_HIGH_PAIRED__condX2 <= 0
 c2912: - 2 aux_cap_helper__b5,r1,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b5,r1,band_HIGH_PAIRED__condX2 <= 0
 c2913: - aux_Omega_b4,r5 + 699.148980775203 aux_sv_b4,r5  = 0
 c2914: - 3 aux_cap_helper__b9,r10,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b9,r10,band_HIGH_PAIRED__condX3 <= 0
 c2915: sv_function_helpervar_b4,r4__condX2
        + sv_function_helpervar_b4,r4__condX1
        + sv_function_helpervar_b4,r4__condX3 - aux_c_b4,r4  = 0
 c2916: - aux_cap_helper__b5,r5,band_LOW_PAIRED__Z1
        + aux_cap_helper__b5,r5,band_LOW_PAIRED__condX1 <= 0
 c2917: aux_cap_helper__b1,r5,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b1,r5,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b1,r5,band_HIGH_PAIRED__Z2  = 1
 c2918: - 15.9946187432626 sv_function_helpervar_b8,r9__Z3
        + sv_function_helpervar_b8,r9__condX3 <= 0
 c2919: - 1.4943850124 aux_cap_helper__b0,r5,band_LOW_PAIRED__Z2
        + 3.735962531 aux_cap_helper__b0,r5,band_LOW_PAIRED__condX1
        + 5.2303475434 aux_cap_helper__b0,r5,band_LOW_PAIRED__condX2
        - aux_cap_b0,r5,band_LOW_PAIRED  = 0
 c2920: - aux_cap_b3,r11,band_UNPAIRED
        - 0.3214355447 aux_cap_helper__b3,r11,band_UNPAIRED__Z2
        + 0.8035888618 aux_cap_helper__b3,r11,band_UNPAIRED__condX1
        + 1.1250244065 aux_cap_helper__b3,r11,band_UNPAIRED__condX2  = 0
 c2921: - 2 aux_cap_helper__b1,r0,band_LOW_PAIRED__Z2
        + aux_cap_helper__b1,r0,band_LOW_PAIRED__condX2 <= 0
 c2922: - aux_cap_helper__b3,r10,band_HIGH_PAIRED__condX1 <= 0
 c2923: sv_function_helpervar_b3,r7__condX2
        - 6.24248591508399 sv_function_helpervar_b3,r7__Z2 <= 0
 c2924: - aux_cap_helper__b1,r0,band_HIGH_PAIRED__condX1 <= 0
 c2925: - aux_cap_helper__b5,r0,band_LOW_PAIRED__condX2
        + aux_cap_helper__b5,r0,band_LOW_PAIRED__Z2 <= 0
 c2926: - 7.72501854783287 sv_function_helpervar_b7,r0__Z2
        + sv_function_helpervar_b7,r0__condX2 <= 0
 c2927: aux_cap_helper__b7,r1,band_UNPAIRED__condX1
        + aux_cap_helper__b7,r1,band_UNPAIRED__condX2 - X__b7,r1,band_UNPAIRED
         = 0
 c2928: - 2 aux_cap_helper__b6,r9,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b6,r9,band_HIGH_PAIRED__condX2 <= 0
 c2929: - aux_cap_helper__b4,r2,band_UNPAIRED__condX1 <= 0
 c2930: 20.6704944627 sv_function_helpervar_b4,r8__condX3
        + 59.9767000562 sv_function_helpervar_b4,r8__condX2
        + 1289.327263405 sv_function_helpervar_b4,r8__condX1
        + 196.62993565129 sv_function_helpervar_b4,r8__Z2
        + 433.172137247066 sv_function_helpervar_b4,r8__Z3 - aux_sv_b4,r8  = 0
 c2931: - aux_cap_helper__b0,r0,band_LOW_PAIRED__Z1
        + aux_cap_helper__b0,r0,band_LOW_PAIRED__condX1 <= 0
 c2932: aux_cap_helper__b2,r7,band_UNPAIRED__Z2
        - aux_cap_helper__b2,r7,band_UNPAIRED__condX2 <= 0
 c2933: - sv_function_helpervar_b2,r8__condX3
        + 6.53881311386621 sv_function_helpervar_b2,r8__Z3 <= 0
 c2934: - 3 aux_cap_helper__b2,r12,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b2,r12,band_HIGH_PAIRED__condX3 <= 0
 c2935: - 2 aux_cap_helper__b0,r0,band_LOW_PAIRED__Z2
        + aux_cap_helper__b0,r0,band_LOW_PAIRED__condX2 <= 0
 c2936: aux_cap_helper__b4,r9,band_UNPAIRED__condX1
        + aux_cap_helper__b4,r9,band_UNPAIRED__condX2 - X__b4,r9,band_UNPAIRED
         = 0
 c2937: - aux_cap_helper__b8,r2,band_HIGH_PAIRED__condX1 <= 0
 c2938: 2 aux_cap_helper__b6,r3,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b6,r3,band_HIGH_PAIRED__condX3 <= 0
 c2939: aux_cap_b1,r11,band_HIGH_PAIRED + aux_cap_b1,r11,band_UNPAIRED
        + aux_cap_b1,r11,band_LOW_PAIRED - aux_c_b1,r11  = 0
 c2940: aux_cap_b9,r5,band_HIGH_PAIRED + aux_cap_b9,r5,band_UNPAIRED
        + aux_cap_b9,r5,band_LOW_PAIRED - aux_c_b9,r5  = 0
 c2941: X__b5,r1,band_LOW_PAIRED + X__b1,r1,band_LOW_PAIRED
        + X__b7,r1,band_LOW_PAIRED + X__b9,r1,band_LOW_PAIRED
        + X__b4,r1,band_LOW_PAIRED + X__b2,r1,band_LOW_PAIRED
        + X__b6,r1,band_LOW_PAIRED + X__b8,r1,band_LOW_PAIRED
        + X__b0,r1,band_LOW_PAIRED + X__b3,r1,band_LOW_PAIRED <= 2
 c2942: 5.2303475434 aux_cap_helper__b7,r4,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b7,r4,band_LOW_PAIRED__condX1
        - aux_cap_b7,r4,band_LOW_PAIRED
        - 1.4943850124 aux_cap_helper__b7,r4,band_LOW_PAIRED__Z2  = 0
 c2943: aux_cap_b9,r9,band_LOW_PAIRED + aux_cap_b9,r9,band_HIGH_PAIRED
        + aux_cap_b9,r9,band_UNPAIRED - aux_c_b9,r9  = 0
 c2944: sv_function_helpervar_b9,r3__condX2
        - 8.11597258834619 sv_function_helpervar_b9,r3__Z2 <= 0
 c2945: - sv_function_helpervar_b7,r11__condX3
        + 6.9028137273107 sv_function_helpervar_b7,r11__Z3 <= 0
 c2946: - aux_cap_helper__b5,r13,band_HIGH_PAIRED__condX1 <= 0
 c2947: - 5.94106823473582 sv_function_helpervar_b0,r6__Z2
        + sv_function_helpervar_b0,r6__condX2 <= 0
 c2948: 2 aux_cap_helper__b4,r12,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b4,r12,band_HIGH_PAIRED__condX3 <= 0
 c2949: aux_cap_helper__b8,r7,band_LOW_PAIRED__Z2
        + aux_cap_helper__b8,r7,band_LOW_PAIRED__Z1  = 1
 c2950: - aux_cap_helper__b8,r10,band_LOW_PAIRED__condX1 <= 0
 c2951: - aux_cap_helper__b8,r4,band_LOW_PAIRED__condX1 <= 0
 c2952: aux_cap_helper__b5,r9,band_UNPAIRED__Z2
        + aux_cap_helper__b5,r9,band_UNPAIRED__Z1  = 1
 c2953: aux_cap_helper__b3,r9,band_UNPAIRED__condX2
        + aux_cap_helper__b3,r9,band_UNPAIRED__condX1 - X__b3,r9,band_UNPAIRED
         = 0
 c2954: 530.59474590988 sv_function_helpervar_b9,r1__Z3
        + 263.973875678994 sv_function_helpervar_b9,r1__Z2 - aux_sv_b9,r1
        + 1719.0945876837 sv_function_helpervar_b9,r1__condX1
        + 68.7027914013 sv_function_helpervar_b9,r1__condX2
        + 30.49684118 sv_function_helpervar_b9,r1__condX3  = 0
 c2955: - aux_cap_helper__b4,r10,band_UNPAIRED__condX1 <= 0
 c2956: aux_cap_helper__b8,r7,band_UNPAIRED__Z2
        - aux_cap_helper__b8,r7,band_UNPAIRED__condX2 <= 0
 c2957: - 2 aux_cap_helper__b3,r9,band_LOW_PAIRED__Z2
        + aux_cap_helper__b3,r9,band_LOW_PAIRED__condX2 <= 0
 c2958: - aux_cap_helper__b5,r10,band_LOW_PAIRED__condX1 <= 0
 c2959: - aux_cap_b1,r8,band_LOW_PAIRED
        - 1.4943850124 aux_cap_helper__b1,r8,band_LOW_PAIRED__Z2
        + 3.735962531 aux_cap_helper__b1,r8,band_LOW_PAIRED__condX1
        + 5.2303475434 aux_cap_helper__b1,r8,band_LOW_PAIRED__condX2  = 0
 c2960: - aux_cap_helper__b6,r9,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b6,r9,band_HIGH_PAIRED__condX1 <= 0
 c2961: sv_function_helpervar_b8,r4__condX2
        - 8.24097916477441 sv_function_helpervar_b8,r4__Z2 <= 0
 c2962: sv_function_helpervar_b7,r2__Z2 + sv_function_helpervar_b7,r2__Z3
        + sv_function_helpervar_b7,r2__Z1  = 1
 c2963: 2 aux_cap_helper__b3,r3,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b3,r3,band_HIGH_PAIRED__condX3 <= 0
 c2964: - 2 aux_cap_helper__b8,r12,band_UNPAIRED__Z2
        + aux_cap_helper__b8,r12,band_UNPAIRED__condX2 <= 0
 c2965: - 3 aux_cap_helper__b3,r6,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b3,r6,band_HIGH_PAIRED__condX3 <= 0
 c2966: - sv_function_helpervar_b9,r5__condX3
        + 7.61006379523071 sv_function_helpervar_b9,r5__Z3 <= 0
 c2967: sv_function_helpervar_b6,r1__condX3
        - 15.9946187432626 sv_function_helpervar_b6,r1__Z3 <= 0
 c2968: - aux_cap_helper__b8,r8,band_HIGH_PAIRED__condX1 <= 0
 c2969: sv_function_helpervar_b6,r12__condX1
        - 0.15994618744279 sv_function_helpervar_b6,r12__Z1 <= 0
 c2970: 0.159946187438242 sv_function_helpervar_b0,r11__Z2
        - sv_function_helpervar_b0,r11__condX2 <= 0
 c2971: - aux_cap_helper__b1,r5,band_LOW_PAIRED__condX1 <= 0
 c2972: aux_cap_helper__b6,r8,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b6,r8,band_UNPAIRED__Z2 <= 0
 c2973: aux_cap_helper__b3,r9,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b3,r9,band_HIGH_PAIRED__condX2 <= 0
 c2974: 3.735962531 aux_cap_helper__b7,r12,band_LOW_PAIRED__condX1
        + 5.2303475434 aux_cap_helper__b7,r12,band_LOW_PAIRED__condX2
        - 1.4943850124 aux_cap_helper__b7,r12,band_LOW_PAIRED__Z2
        - aux_cap_b7,r12,band_LOW_PAIRED  = 0
 c2975: - aux_cap_helper__b0,r4,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b0,r4,band_HIGH_PAIRED__Z2 <= 0
 c2976: aux_cap_helper__b2,r3,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b2,r3,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b2,r3,band_HIGH_PAIRED__condX1
        - X__b2,r3,band_HIGH_PAIRED  = 0
 c2977: aux_cap_helper__b7,r13,band_LOW_PAIRED__Z1
        + aux_cap_helper__b7,r13,band_LOW_PAIRED__Z2  = 1
 c2978: aux_cap_helper__b6,r5,band_LOW_PAIRED__condX1
        - aux_cap_helper__b6,r5,band_LOW_PAIRED__Z1 <= 0
 c2979: aux_cap_helper__b7,r5,band_LOW_PAIRED__condX2
        + aux_cap_helper__b7,r5,band_LOW_PAIRED__condX1
        - X__b7,r5,band_LOW_PAIRED  = 0
 c2980: aux_cap_helper__b9,r1,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b9,r1,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b9,r1,band_HIGH_PAIRED__condX3
        - X__b9,r1,band_HIGH_PAIRED  = 0
 c2981: aux_cap_helper__b9,r6,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b9,r6,band_HIGH_PAIRED__Z3 <= 0
 c2982: - aux_cap_helper__b0,r1,band_UNPAIRED__Z1
        + aux_cap_helper__b0,r1,band_UNPAIRED__condX1 <= 0
 c2983: 35.1747103793752 aux_sv_b2,r13 - aux_Omega_b2,r13  = 0
 c2984: 2 aux_cap_helper__b3,r9,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b3,r9,band_HIGH_PAIRED__condX3 <= 0
 c2985: - aux_cap_helper__b0,r4,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b0,r4,band_HIGH_PAIRED__Z3 <= 0
 c2986: - aux_cap_helper__b0,r12,band_HIGH_PAIRED__condX1 <= 0
 c2987: - aux_cap_helper__b9,r12,band_LOW_PAIRED__condX1 <= 0
 c2988: sv_function_helpervar_b6,r1__condX3
        + sv_function_helpervar_b6,r1__condX2
        + sv_function_helpervar_b6,r1__condX1 - aux_c_b6,r1  = 0
 c2989: - 1.4943850124 aux_cap_helper__b8,r7,band_LOW_PAIRED__Z2
        - aux_cap_b8,r7,band_LOW_PAIRED
        + 5.2303475434 aux_cap_helper__b8,r7,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b8,r7,band_LOW_PAIRED__condX1  = 0
 c2990: aux_cap_helper__b6,r5,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b6,r5,band_LOW_PAIRED__Z2 <= 0
 c2991: - aux_cap_helper__b3,r6,band_UNPAIRED__condX2
        + aux_cap_helper__b3,r6,band_UNPAIRED__Z2 <= 0
 c2992: sv_function_helpervar_b4,r13__condX2
        - 6.03557477569176 sv_function_helpervar_b4,r13__Z2 <= 0
 c2993: aux_cap_helper__b6,r6,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b6,r6,band_HIGH_PAIRED__Z3 <= 0
 c2994: - aux_cap_helper__b8,r12,band_UNPAIRED__Z1
        + aux_cap_helper__b8,r12,band_UNPAIRED__condX1 <= 0
 c2995: aux_cap_helper__b9,r9,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b9,r9,band_HIGH_PAIRED__Z2 <= 0
 c2996: - 0.3214355447 aux_cap_helper__b4,r7,band_UNPAIRED__Z2
        + 0.8035888618 aux_cap_helper__b4,r7,band_UNPAIRED__condX1
        + 1.1250244065 aux_cap_helper__b4,r7,band_UNPAIRED__condX2
        - aux_cap_b4,r7,band_UNPAIRED  = 0
 c2997: 1.1250244065 aux_cap_helper__b7,r9,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b7,r9,band_UNPAIRED__condX1
        - 0.3214355447 aux_cap_helper__b7,r9,band_UNPAIRED__Z2
        - aux_cap_b7,r9,band_UNPAIRED  = 0
 c2998: sv_function_helpervar_b3,r7__condX3
        - 15.9946187432626 sv_function_helpervar_b3,r7__Z3 <= 0
 c2999: aux_cap_helper__b6,r6,band_LOW_PAIRED__condX2
        + aux_cap_helper__b6,r6,band_LOW_PAIRED__condX1
        - X__b6,r6,band_LOW_PAIRED  = 0
 c3000: - aux_cap_b8,r13,band_UNPAIRED
        + 0.8035888618 aux_cap_helper__b8,r13,band_UNPAIRED__condX1
        + 1.1250244065 aux_cap_helper__b8,r13,band_UNPAIRED__condX2
        - 0.3214355447 aux_cap_helper__b8,r13,band_UNPAIRED__Z2  = 0
 c3001: aux_cap_helper__b2,r0,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b2,r0,band_UNPAIRED__Z2 <= 0
 c3002: - aux_cap_helper__b3,r3,band_UNPAIRED__condX1 <= 0
 c3003: aux_cap_helper__b1,r13,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b1,r13,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b1,r13,band_HIGH_PAIRED__condX3
        - X__b1,r13,band_HIGH_PAIRED  = 0
 c3004: - X__b1,r8,band_HIGH_PAIRED
        + aux_cap_helper__b1,r8,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b1,r8,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b1,r8,band_HIGH_PAIRED__condX3  = 0
 c3005: - sv_function_helpervar_b5,r13__condX1 <= 0
 c3006: - aux_cap_helper__b2,r10,band_LOW_PAIRED__condX1 <= 0
 c3007: - sv_function_helpervar_b9,r9__condX1 <= 0
 c3008: - 2 aux_cap_helper__b4,r9,band_UNPAIRED__Z2
        + aux_cap_helper__b4,r9,band_UNPAIRED__condX2 <= 0
 c3009: sv_function_helpervar_b0,r13__Z2 + sv_function_helpervar_b0,r13__Z1
        + sv_function_helpervar_b0,r13__Z3  = 1
 c3010: - 2 aux_cap_helper__b5,r2,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b5,r2,band_HIGH_PAIRED__condX2 <= 0
 c3011: sv_function_helpervar_b6,r3__condX3
        - 15.9946187432626 sv_function_helpervar_b6,r3__Z3 <= 0
 c3012: aux_cap_helper__b3,r4,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b3,r4,band_HIGH_PAIRED__condX2 <= 0
 c3013: - aux_cap_helper__b4,r11,band_LOW_PAIRED__condX1 <= 0
 c3014: - X__b9,r10,band_HIGH_PAIRED
        + aux_cap_helper__b9,r10,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b9,r10,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b9,r10,band_HIGH_PAIRED__condX3  = 0
 c3015: - sv_function_helpervar_b8,r8__condX2
        + 0.159946187496666 sv_function_helpervar_b8,r8__Z2 <= 0
 c3016: - X__b1,r0,band_LOW_PAIRED
        + aux_cap_helper__b1,r0,band_LOW_PAIRED__condX1
        + aux_cap_helper__b1,r0,band_LOW_PAIRED__condX2  = 0
 c3017: - sv_function_helpervar_b1,r2__condX1 <= 0
 c3018: - aux_Omega_b0,r12 + 990.441975492583 aux_sv_b0,r12  = 0
 c3019: aux_cap_helper__b6,r8,band_UNPAIRED__Z2
        + aux_cap_helper__b6,r8,band_UNPAIRED__Z1  = 1
 c3020: 427.925979611511 sv_function_helpervar_b4,r7__Z3
        + 196.878518580318 sv_function_helpervar_b4,r7__Z2
        + 58.4225340354 sv_function_helpervar_b4,r7__condX2
        + 20.9984896291 sv_function_helpervar_b4,r7__condX3
        + 1289.3272633111 sv_function_helpervar_b4,r7__condX1 - aux_sv_b4,r7
         = 0
 c3021: 24.4496686962855 sv_function_helpervar_b2,r5__Z2
        + 55.4809564391114 sv_function_helpervar_b2,r5__Z3
        + 7.9749282674 sv_function_helpervar_b2,r5__condX2
        + 2.4881909769 sv_function_helpervar_b2,r5__condX3
        + 160.836769375 sv_function_helpervar_b2,r5__condX1 - aux_sv_b2,r5  = 0
 c3022: aux_cap_b4,r5,band_UNPAIRED + aux_cap_b4,r5,band_LOW_PAIRED
        + aux_cap_b4,r5,band_HIGH_PAIRED - aux_c_b4,r5  = 0
 c3023: - aux_cap_b0,r3,band_LOW_PAIRED
        + 3.735962531 aux_cap_helper__b0,r3,band_LOW_PAIRED__condX1
        + 5.2303475434 aux_cap_helper__b0,r3,band_LOW_PAIRED__condX2
        - 1.4943850124 aux_cap_helper__b0,r3,band_LOW_PAIRED__Z2  = 0
 c3024: X__b7,r2,band_HIGH_PAIRED - GlobalBidder_W_ir__b7,r2
        + X__b7,r2,band_LOW_PAIRED + X__b7,r2,band_UNPAIRED >= 0
 c3025: aux_cap_helper__b8,r1,band_LOW_PAIRED__condX1
        - aux_cap_helper__b8,r1,band_LOW_PAIRED__Z1 <= 0
 c3026: aux_cap_helper__b3,r11,band_UNPAIRED__Z2
        - aux_cap_helper__b3,r11,band_UNPAIRED__condX2 <= 0
 c3027: - sv_function_helpervar_b5,r10__condX3
        + 5.86499039658243 sv_function_helpervar_b5,r10__Z3 <= 0
 c3028: - 2 aux_cap_helper__b5,r12,band_UNPAIRED__Z2
        + aux_cap_helper__b5,r12,band_UNPAIRED__condX2 <= 0
 c3029: - aux_cap_helper__b8,r0,band_UNPAIRED__condX1 <= 0
 c3030: - sv_function_helpervar_b6,r4__condX1 <= 0
 c3031: aux_cap_helper__b3,r12,band_LOW_PAIRED__Z1
        + aux_cap_helper__b3,r12,band_LOW_PAIRED__Z2  = 1
 c3032: - 2 aux_cap_helper__b0,r1,band_UNPAIRED__Z2
        + aux_cap_helper__b0,r1,band_UNPAIRED__condX2 <= 0
 c3033: 0.0102040816326531 X__b8,r8,band_HIGH_PAIRED
        + 0.0102040816326531 X__b8,r8,band_LOW_PAIRED
        - GlobalBidder_W_ir__b8,r8 + 0.0102040816326531 X__b8,r8,band_UNPAIRED
        <= 0
 c3034: - 15.9946187432626 sv_function_helpervar_b9,r8__Z3
        + sv_function_helpervar_b9,r8__condX3 <= 0
 c3035: aux_cap_helper__b4,r6,band_LOW_PAIRED__Z2
        - aux_cap_helper__b4,r6,band_LOW_PAIRED__condX2 <= 0
 c3036: - aux_cap_helper__b9,r6,band_UNPAIRED__condX2
        + aux_cap_helper__b9,r6,band_UNPAIRED__Z2 <= 0
 c3037: - aux_cap_helper__b4,r9,band_UNPAIRED__Z1
        + aux_cap_helper__b4,r9,band_UNPAIRED__condX1 <= 0
 c3038: - aux_cap_helper__b5,r2,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b5,r2,band_HIGH_PAIRED__condX1 <= 0
 c3039: aux_cap_helper__b6,r8,band_UNPAIRED__condX1
        - aux_cap_helper__b6,r8,band_UNPAIRED__Z1 <= 0
 c3040: aux_cap_helper__b5,r4,band_LOW_PAIRED__Z2
        + aux_cap_helper__b5,r4,band_LOW_PAIRED__Z1  = 1
 c3041: X__b7,r12,band_HIGH_PAIRED + X__b7,r12,band_UNPAIRED
        + X__b7,r12,band_LOW_PAIRED - GlobalBidder_W_ir__b7,r12 >= 0
 c3042: - aux_cap_helper__b7,r11,band_LOW_PAIRED__condX1 <= 0
 c3043: aux_cap_helper__b2,r2,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b2,r2,band_HIGH_PAIRED__Z1 <= 0
 c3044: - aux_cap_helper__b4,r0,band_HIGH_PAIRED__condX1 <= 0
 c3045: aux_cap_helper__b9,r13,band_UNPAIRED__Z1
        + aux_cap_helper__b9,r13,band_UNPAIRED__Z2  = 1
 c3046: aux_cap_helper__b8,r11,band_UNPAIRED__condX2
        + aux_cap_helper__b8,r11,band_UNPAIRED__condX1
        - X__b8,r11,band_UNPAIRED  = 0
 c3047: - aux_sv_b0,r2 + 1.8401812166 sv_function_helpervar_b0,r2__condX3
        + 109.8706625611 sv_function_helpervar_b0,r2__condX1
        + 4.7638095725 sv_function_helpervar_b0,r2__condX2
        + 16.8114404162851 sv_function_helpervar_b0,r2__Z2
        + 35.6536459690043 sv_function_helpervar_b0,r2__Z3  = 0
 c3048: - aux_cap_helper__b5,r8,band_LOW_PAIRED__Z1
        + aux_cap_helper__b5,r8,band_LOW_PAIRED__condX1 <= 0
 c3049: - X__b9,r5,band_LOW_PAIRED
        + aux_cap_helper__b9,r5,band_LOW_PAIRED__condX2
        + aux_cap_helper__b9,r5,band_LOW_PAIRED__condX1  = 0
 c3050: - aux_cap_helper__b5,r12,band_UNPAIRED__Z1
        + aux_cap_helper__b5,r12,band_UNPAIRED__condX1 <= 0
 c3051: - 15.9946187432626 sv_function_helpervar_b1,r1__Z3
        + sv_function_helpervar_b1,r1__condX3 <= 0
 c3052: 0.159946187454999 sv_function_helpervar_b7,r9__Z2
        - sv_function_helpervar_b7,r9__condX2 <= 0
 c3053: sv_function_helpervar_b5,r1__condX1
        - 0.159946187453604 sv_function_helpervar_b5,r1__Z1 <= 0
 c3054: - aux_cap_helper__b6,r1,band_UNPAIRED__condX1 <= 0
 c3055: - sv_function_helpervar_b9,r0__condX1 <= 0
 c3056: sv_function_helpervar_b3,r13__condX2
        - 5.70920988297092 sv_function_helpervar_b3,r13__Z2 <= 0
 c3057: aux_cap_helper__b4,r12,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b4,r12,band_HIGH_PAIRED__condX2 <= 0
 c3058: - 2 aux_cap_helper__b0,r11,band_UNPAIRED__Z2
        + aux_cap_helper__b0,r11,band_UNPAIRED__condX2 <= 0
 c3059: - 1.4943850124 aux_cap_helper__b1,r6,band_LOW_PAIRED__Z2
        - aux_cap_b1,r6,band_LOW_PAIRED
        + 3.735962531 aux_cap_helper__b1,r6,band_LOW_PAIRED__condX1
        + 5.2303475434 aux_cap_helper__b1,r6,band_LOW_PAIRED__condX2  = 0
 c3060: sv_function_helpervar_b7,r3__Z1 + sv_function_helpervar_b7,r3__Z2
        + sv_function_helpervar_b7,r3__Z3  = 1
 c3061: - 6.53655273965809 sv_function_helpervar_b7,r4__Z2
        + sv_function_helpervar_b7,r4__condX2 <= 0
 c3062: 431.668741911943 sv_function_helpervar_b8,r5__Z3
        + 205.653787454493 sv_function_helpervar_b8,r5__Z2 - aux_sv_b8,r5
        + 1343.0085755749 sv_function_helpervar_b8,r5__condX1
        + 57.2399632965 sv_function_helpervar_b8,r5__condX2
        + 22.7526850248 sv_function_helpervar_b8,r5__condX3  = 0
 c3063: - 2 aux_cap_helper__b5,r8,band_LOW_PAIRED__Z2
        + aux_cap_helper__b5,r8,band_LOW_PAIRED__condX2 <= 0
 c3064: aux_cap_helper__b2,r3,band_UNPAIRED__condX2
        + aux_cap_helper__b2,r3,band_UNPAIRED__condX1 - X__b2,r3,band_UNPAIRED
         = 0
 c3065: - sv_function_helpervar_b4,r9__condX1 <= 0
 c3066: aux_cap_b9,r13,band_HIGH_PAIRED + aux_cap_b9,r13,band_LOW_PAIRED
        + aux_cap_b9,r13,band_UNPAIRED - aux_c_b9,r13  = 0
 c3067: - aux_cap_helper__b3,r9,band_LOW_PAIRED__Z1
        + aux_cap_helper__b3,r9,band_LOW_PAIRED__condX1 <= 0
 c3068: - aux_c_b6,r12 + sv_function_helpervar_b6,r12__condX1
        + sv_function_helpervar_b6,r12__condX2
        + sv_function_helpervar_b6,r12__condX3  = 0
 c3069: aux_cap_helper__b5,r13,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b5,r13,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b5,r13,band_HIGH_PAIRED__Z2  = 1
 c3070: 6.3832954915594 sv_function_helpervar_b5,r13__Z3
        - sv_function_helpervar_b5,r13__condX3 <= 0
 c3071: - 7.03795147765963 sv_function_helpervar_b8,r1__Z2
        + sv_function_helpervar_b8,r1__condX2 <= 0
 c3072: - 15.9946187432626 sv_function_helpervar_b5,r12__Z3
        + sv_function_helpervar_b5,r12__condX3 <= 0
 c3073: aux_cap_helper__b2,r12,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b2,r12,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b2,r12,band_HIGH_PAIRED__Z1  = 1
 c3074: aux_cap_helper__b8,r3,band_UNPAIRED__Z1
        + aux_cap_helper__b8,r3,band_UNPAIRED__Z2  = 1
 c3075: aux_cap_helper__b2,r4,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b2,r4,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b2,r4,band_HIGH_PAIRED__Z3  = 1
 c3076: - aux_cap_b1,r9,band_UNPAIRED
        - 0.3214355447 aux_cap_helper__b1,r9,band_UNPAIRED__Z2
        + 1.1250244065 aux_cap_helper__b1,r9,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b1,r9,band_UNPAIRED__condX1  = 0
 c3077: aux_cap_helper__b2,r0,band_UNPAIRED__condX1
        - aux_cap_helper__b2,r0,band_UNPAIRED__Z1 <= 0
 c3078: 1.1250244065 aux_cap_helper__b9,r13,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b9,r13,band_UNPAIRED__condX1
        - 0.3214355447 aux_cap_helper__b9,r13,band_UNPAIRED__Z2
        - aux_cap_b9,r13,band_UNPAIRED  = 0
 c3079: aux_cap_helper__b0,r8,band_UNPAIRED__Z2
        - aux_cap_helper__b0,r8,band_UNPAIRED__condX2 <= 0
 c3080: aux_cap_helper__b2,r1,band_LOW_PAIRED__Z1
        + aux_cap_helper__b2,r1,band_LOW_PAIRED__Z2  = 1
 c3081: - sv_function_helpervar_b5,r9__condX3
        + 6.12746312836256 sv_function_helpervar_b5,r9__Z3 <= 0
 c3082: 5.78594863013166 sv_function_helpervar_b3,r12__Z3
        - sv_function_helpervar_b3,r12__condX3 <= 0
 c3083: - 2 aux_cap_helper__b3,r6,band_LOW_PAIRED__Z2
        + aux_cap_helper__b3,r6,band_LOW_PAIRED__condX2 <= 0
 c3084: aux_cap_helper__b7,r5,band_UNPAIRED__Z2
        - aux_cap_helper__b7,r5,band_UNPAIRED__condX2 <= 0
 c3085: aux_cap_b0,r8,band_HIGH_PAIRED + aux_cap_b0,r8,band_UNPAIRED
        - aux_c_b0,r8 + aux_cap_b0,r8,band_LOW_PAIRED  = 0
 c3086: - 2 aux_cap_helper__b0,r9,band_UNPAIRED__Z2
        + aux_cap_helper__b0,r9,band_UNPAIRED__condX2 <= 0
 c3087: - aux_cap_helper__b5,r8,band_UNPAIRED__condX2
        + aux_cap_helper__b5,r8,band_UNPAIRED__Z2 <= 0
 c3088: aux_cap_b2,r4,band_HIGH_PAIRED - aux_c_b2,r4
        + aux_cap_b2,r4,band_LOW_PAIRED + aux_cap_b2,r4,band_UNPAIRED  = 0
 c3089: - 2 aux_cap_helper__b2,r8,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b2,r8,band_HIGH_PAIRED__condX2 <= 0
 c3090: - 2 aux_cap_helper__b6,r0,band_UNPAIRED__Z2
        + aux_cap_helper__b6,r0,band_UNPAIRED__condX2 <= 0
 c3091: aux_cap_helper__b4,r4,band_UNPAIRED__condX2
        + aux_cap_helper__b4,r4,band_UNPAIRED__condX1 - X__b4,r4,band_UNPAIRED
         = 0
 c3092: aux_cap_helper__b0,r11,band_UNPAIRED__Z1
        + aux_cap_helper__b0,r11,band_UNPAIRED__Z2  = 1
 c3093: - 5.54618304423859 sv_function_helpervar_b5,r12__Z2
        + sv_function_helpervar_b5,r12__condX2 <= 0
 c3094: 0.0102040816326531 X__b7,r9,band_LOW_PAIRED - GlobalBidder_W_ir__b7,r9
        + 0.0102040816326531 X__b7,r9,band_UNPAIRED
        + 0.0102040816326531 X__b7,r9,band_HIGH_PAIRED <= 0
 c3095: - 2 aux_cap_helper__b4,r12,band_LOW_PAIRED__Z2
        + aux_cap_helper__b4,r12,band_LOW_PAIRED__condX2 <= 0
 c3096: aux_cap_helper__b3,r12,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b3,r12,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b3,r12,band_HIGH_PAIRED__Z1  = 1
 c3097: - aux_cap_helper__b2,r7,band_LOW_PAIRED__condX2
        + aux_cap_helper__b2,r7,band_LOW_PAIRED__Z2 <= 0
 c3098: - aux_cap_helper__b2,r6,band_UNPAIRED__condX2
        + aux_cap_helper__b2,r6,band_UNPAIRED__Z2 <= 0
 c3099: - sv_function_helpervar_b6,r6__condX1 <= 0
 c3100: 0.927217345653617 aux_cap_helper__b8,r10,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b8,r10,band_HIGH_PAIRED__Z2
        + 1.3908260184 aux_cap_helper__b8,r10,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b8,r10,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b8,r10,band_HIGH_PAIRED__condX1
        - aux_cap_b8,r10,band_HIGH_PAIRED  = 0
 c3101: - 3 aux_cap_helper__b4,r10,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b4,r10,band_HIGH_PAIRED__condX3 <= 0
 c3102: - aux_cap_helper__b4,r1,band_HIGH_PAIRED__condX1 <= 0
 c3103: aux_cap_helper__b4,r5,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b4,r5,band_LOW_PAIRED__Z2 <= 0
 c3104: aux_cap_helper__b4,r5,band_LOW_PAIRED__condX1
        - aux_cap_helper__b4,r5,band_LOW_PAIRED__Z1 <= 0
 c3105: - aux_cap_helper__b7,r13,band_LOW_PAIRED__condX2
        + aux_cap_helper__b7,r13,band_LOW_PAIRED__Z2 <= 0
 c3106: aux_cap_helper__b2,r0,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b2,r0,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b2,r0,band_HIGH_PAIRED__condX2
        - X__b2,r0,band_HIGH_PAIRED  = 0
 c3107: - aux_cap_helper__b0,r2,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b0,r2,band_HIGH_PAIRED__Z2 <= 0
 c3108: - aux_cap_helper__b4,r3,band_UNPAIRED__condX1 <= 0
 c3109: - aux_c_b0,r13 + sv_function_helpervar_b0,r13__condX1
        + sv_function_helpervar_b0,r13__condX2
        + sv_function_helpervar_b0,r13__condX3  = 0
 c3110: - sv_function_helpervar_b8,r2__condX1 <= 0
 c3111: - aux_cap_helper__b2,r0,band_LOW_PAIRED__condX2
        + aux_cap_helper__b2,r0,band_LOW_PAIRED__Z2 <= 0
 c3112: aux_cap_helper__b0,r1,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b0,r1,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b0,r1,band_HIGH_PAIRED__Z2  = 1
 c3113: - sv_function_helpervar_b5,r8__condX1 <= 0
 c3114: - X__b9,r8,band_LOW_PAIRED
        + aux_cap_helper__b9,r8,band_LOW_PAIRED__condX2
        + aux_cap_helper__b9,r8,band_LOW_PAIRED__condX1  = 0
 c3115: - aux_cap_helper__b1,r11,band_LOW_PAIRED__condX1 <= 0
 c3116: - aux_cap_helper__b0,r2,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b0,r2,band_HIGH_PAIRED__Z3 <= 0
 c3117: - aux_cap_helper__b3,r6,band_LOW_PAIRED__Z1
        + aux_cap_helper__b3,r6,band_LOW_PAIRED__condX1 <= 0
 c3118: 1.5453622427 aux_cap_helper__b3,r10,band_HIGH_PAIRED__condX1
        + 2.1635071398 aux_cap_helper__b3,r10,band_HIGH_PAIRED__condX2
        + 1.3908260184 aux_cap_helper__b3,r10,band_HIGH_PAIRED__condX3
        - aux_cap_b3,r10,band_HIGH_PAIRED
        - 0.6181448971 aux_cap_helper__b3,r10,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b3,r10,band_HIGH_PAIRED__Z3  = 0
 c3119: - aux_cap_helper__b0,r9,band_UNPAIRED__Z1
        + aux_cap_helper__b0,r9,band_UNPAIRED__condX1 <= 0
 c3120: - aux_cap_helper__b4,r12,band_LOW_PAIRED__Z1
        + aux_cap_helper__b4,r12,band_LOW_PAIRED__condX1 <= 0
 c3121: aux_cap_helper__b9,r4,band_LOW_PAIRED__Z2
        + aux_cap_helper__b9,r4,band_LOW_PAIRED__Z1  = 1
 c3122: aux_cap_helper__b9,r4,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b9,r4,band_HIGH_PAIRED__Z1 <= 0
 c3123: - aux_cap_helper__b0,r12,band_LOW_PAIRED__condX1 <= 0
 c3124: - X__b3,r12,band_HIGH_PAIRED
        + aux_cap_helper__b3,r12,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b3,r12,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b3,r12,band_HIGH_PAIRED__condX1  = 0
 c3125: - 6.12424674439599 sv_function_helpervar_b5,r2__Z2
        + sv_function_helpervar_b5,r2__condX2 <= 0
 c3126: - aux_cap_helper__b1,r8,band_LOW_PAIRED__condX1 <= 0
 c3127: - sv_function_helpervar_b3,r10__condX3
        + 5.97509000744268 sv_function_helpervar_b3,r10__Z3 <= 0
 c3128: - X__b2,r11,band_HIGH_PAIRED
        + aux_cap_helper__b2,r11,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b2,r11,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b2,r11,band_HIGH_PAIRED__condX1  = 0
 c3129: 161.176675753213 sv_function_helpervar_b6,r0__Z2
        + 363.421511527063 sv_function_helpervar_b6,r0__Z3
        + 16.5182124446 sv_function_helpervar_b6,r0__condX3
        + 51.7787281577 sv_function_helpervar_b6,r0__condX2
        + 1059.471867559 sv_function_helpervar_b6,r0__condX1 - aux_sv_b6,r0
         = 0
 c3130: aux_cap_helper__b9,r4,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b9,r4,band_HIGH_PAIRED__Z2 <= 0
 c3131: - 0.3214355447 aux_cap_helper__b9,r1,band_UNPAIRED__Z2
        - aux_cap_b9,r1,band_UNPAIRED
        + 0.8035888618 aux_cap_helper__b9,r1,band_UNPAIRED__condX1
        + 1.1250244065 aux_cap_helper__b9,r1,band_UNPAIRED__condX2  = 0
 c3132: aux_cap_helper__b3,r13,band_LOW_PAIRED__condX1
        - aux_cap_helper__b3,r13,band_LOW_PAIRED__Z1 <= 0
 c3133: aux_cap_helper__b3,r5,band_UNPAIRED__condX1
        + aux_cap_helper__b3,r5,band_UNPAIRED__condX2 - X__b3,r5,band_UNPAIRED
         = 0
 c3134: aux_cap_helper__b6,r7,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b6,r7,band_HIGH_PAIRED__condX2 <= 0
 c3135: - aux_cap_helper__b8,r1,band_LOW_PAIRED__condX1 <= 0
 c3136: - aux_c_b6,r4 + sv_function_helpervar_b6,r4__condX1
        + sv_function_helpervar_b6,r4__condX2
        + sv_function_helpervar_b6,r4__condX3  = 0
 c3137: - aux_cap_b6,r12,band_LOW_PAIRED
        - 1.4943850124 aux_cap_helper__b6,r12,band_LOW_PAIRED__Z2
        + 5.2303475434 aux_cap_helper__b6,r12,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b6,r12,band_LOW_PAIRED__condX1  = 0
 c3138: - 1.4943850124 aux_cap_helper__b4,r7,band_LOW_PAIRED__Z2
        + 5.2303475434 aux_cap_helper__b4,r7,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b4,r7,band_LOW_PAIRED__condX1
        - aux_cap_b4,r7,band_LOW_PAIRED  = 0
 c3139: aux_cap_helper__b3,r12,band_UNPAIRED__condX1
        - aux_cap_helper__b3,r12,band_UNPAIRED__Z1 <= 0
 c3140: aux_cap_helper__b8,r0,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b8,r0,band_HIGH_PAIRED__Z1 <= 0
 c3141: aux_cap_helper__b3,r13,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b3,r13,band_LOW_PAIRED__Z2 <= 0
 c3142: sv_function_helpervar_b7,r3__condX2
        - 7.00344924003967 sv_function_helpervar_b7,r3__Z2 <= 0
 c3143: - aux_cap_helper__b0,r13,band_UNPAIRED__condX1 <= 0
 c3144: - aux_cap_b4,r12,band_UNPAIRED
        - 0.3214355447 aux_cap_helper__b4,r12,band_UNPAIRED__Z2
        + 0.8035888618 aux_cap_helper__b4,r12,band_UNPAIRED__condX1
        + 1.1250244065 aux_cap_helper__b4,r12,band_UNPAIRED__condX2  = 0
 c3145: - 2 aux_cap_helper__b5,r5,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b5,r5,band_HIGH_PAIRED__condX2 <= 0
 c3146: - sv_function_helpervar_b5,r5__condX1 <= 0
 c3147: - X__b5,r4,band_LOW_PAIRED
        + aux_cap_helper__b5,r4,band_LOW_PAIRED__condX2
        + aux_cap_helper__b5,r4,band_LOW_PAIRED__condX1  = 0
 c3148: - aux_c_b7,r10 + aux_cap_b7,r10,band_HIGH_PAIRED
        + aux_cap_b7,r10,band_UNPAIRED + aux_cap_b7,r10,band_LOW_PAIRED  = 0
 c3149: - aux_cap_helper__b2,r8,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b2,r8,band_HIGH_PAIRED__condX1 <= 0
 c3150: - aux_cap_helper__b6,r0,band_UNPAIRED__Z1
        + aux_cap_helper__b6,r0,band_UNPAIRED__condX1 <= 0
 c3151: - X__b0,r6,band_LOW_PAIRED
        + aux_cap_helper__b0,r6,band_LOW_PAIRED__condX2
        + aux_cap_helper__b0,r6,band_LOW_PAIRED__condX1  = 0
 c3152: sv_function_helpervar_b1,r6__condX1
        - 0.159946187453889 sv_function_helpervar_b1,r6__Z1 <= 0
 c3153: - GlobalBidder_W_ir__b9,r6 + 0.0102040816326531 X__b9,r6,band_UNPAIRED
        + 0.0102040816326531 X__b9,r6,band_HIGH_PAIRED
        + 0.0102040816326531 X__b9,r6,band_LOW_PAIRED <= 0
 c3154: - 3 aux_cap_helper__b0,r9,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b0,r9,band_HIGH_PAIRED__condX3 <= 0
 c3155: 5.94064266021643 sv_function_helpervar_b5,r3__Z3
        - sv_function_helpervar_b5,r3__condX3 <= 0
 c3156: aux_cap_helper__b9,r2,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b9,r2,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b9,r2,band_HIGH_PAIRED__Z1  = 1
 c3157: aux_cap_helper__b6,r8,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b6,r8,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b6,r8,band_HIGH_PAIRED__condX1
        - X__b6,r8,band_HIGH_PAIRED  = 0
 c3158: - X__b4,r13,band_HIGH_PAIRED
        + aux_cap_helper__b4,r13,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b4,r13,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b4,r13,band_HIGH_PAIRED__condX2  = 0
 c3159: aux_cap_helper__b2,r1,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b2,r1,band_HIGH_PAIRED__Z3 <= 0
 c3160: aux_cap_helper__b5,r10,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b5,r10,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b5,r10,band_HIGH_PAIRED__Z2  = 1
 c3161: aux_cap_helper__b9,r4,band_UNPAIRED__Z2
        + aux_cap_helper__b9,r4,band_UNPAIRED__Z1  = 1
 c3162: 0.88947 GlobalBidder_PSI__i8,k2 + 0.97237 GlobalBidder_PSI__i8,k1
        + GlobalBidder_PSI__i8,k0 - v__8  = 0
 c3163: - 3 aux_cap_helper__b3,r4,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b3,r4,band_HIGH_PAIRED__condX3 <= 0
 c3164: aux_cap_helper__b4,r8,band_UNPAIRED__condX1
        - aux_cap_helper__b4,r8,band_UNPAIRED__Z1 <= 0
 c3165: sv_function_helpervar_b8,r10__condX3
        + sv_function_helpervar_b8,r10__condX2
        + sv_function_helpervar_b8,r10__condX1 - aux_c_b8,r10  = 0
 c3166: - 2 aux_cap_helper__b9,r13,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b9,r13,band_HIGH_PAIRED__condX2 <= 0
 c3167: - aux_cap_helper__b5,r10,band_UNPAIRED__Z1
        + aux_cap_helper__b5,r10,band_UNPAIRED__condX1 <= 0
 c3168: aux_cap_helper__b2,r7,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b2,r7,band_HIGH_PAIRED__Z1 <= 0
 c3169: sv_function_helpervar_b2,r0__condX3
        - 15.9946187432626 sv_function_helpervar_b2,r0__Z3 <= 0
 c3170: - aux_cap_helper__b1,r12,band_UNPAIRED__condX2
        + aux_cap_helper__b1,r12,band_UNPAIRED__Z2 <= 0
 c3171: - X__b7,r13,band_UNPAIRED
        + aux_cap_helper__b7,r13,band_UNPAIRED__condX2
        + aux_cap_helper__b7,r13,band_UNPAIRED__condX1  = 0
 c3172: 5.746806695232 sv_function_helpervar_b0,r9__Z3
        - sv_function_helpervar_b0,r9__condX3 <= 0
 c3173: - aux_cap_helper__b5,r11,band_LOW_PAIRED__Z1
        + aux_cap_helper__b5,r11,band_LOW_PAIRED__condX1 <= 0
 c3174: - 5.48061618636332 sv_function_helpervar_b5,r7__Z2
        + sv_function_helpervar_b5,r7__condX2 <= 0
 c3175: aux_cap_helper__b2,r7,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b2,r7,band_HIGH_PAIRED__Z2 <= 0
 c3176: - aux_Omega_b4,r7 + 2621.55213359136 aux_sv_b4,r7  = 0
 c3177: aux_cap_helper__b7,r1,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b7,r1,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b7,r1,band_HIGH_PAIRED__Z1  = 1
 c3178: aux_cap_helper__b8,r4,band_LOW_PAIRED__Z2
        + aux_cap_helper__b8,r4,band_LOW_PAIRED__Z1  = 1
 c3179: - X__b9,r1,band_LOW_PAIRED
        + aux_cap_helper__b9,r1,band_LOW_PAIRED__condX2
        + aux_cap_helper__b9,r1,band_LOW_PAIRED__condX1  = 0
 c3180: 2 aux_cap_helper__b6,r7,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b6,r7,band_HIGH_PAIRED__condX3 <= 0
 c3181: - 2 aux_cap_helper__b5,r11,band_LOW_PAIRED__Z2
        + aux_cap_helper__b5,r11,band_LOW_PAIRED__condX2 <= 0
 c3182: - 0.159946187436557 sv_function_helpervar_b6,r0__Z1
        + sv_function_helpervar_b6,r0__condX1 <= 0
 c3183: aux_cap_helper__b3,r12,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b3,r12,band_UNPAIRED__Z2 <= 0
 c3184: aux_cap_helper__b9,r3,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b9,r3,band_HIGH_PAIRED__Z2 <= 0
 c3185: 5.65569045862011 sv_function_helpervar_b2,r5__Z3
        - sv_function_helpervar_b2,r5__condX3 <= 0
 c3186: - sv_function_helpervar_b4,r7__condX1 <= 0
 c3187: sv_function_helpervar_b3,r2__Z1 + sv_function_helpervar_b3,r2__Z3
        + sv_function_helpervar_b3,r2__Z2  = 1
 c3188: sv_function_helpervar_b3,r6__condX1
        + sv_function_helpervar_b3,r6__condX2
        + sv_function_helpervar_b3,r6__condX3 - aux_c_b3,r6  = 0
 c3189: aux_cap_helper__b7,r10,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b7,r10,band_HIGH_PAIRED__Z3 <= 0
 c3190: - aux_cap_helper__b9,r13,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b9,r13,band_HIGH_PAIRED__condX1 <= 0
 c3191: - aux_cap_helper__b8,r4,band_LOW_PAIRED__Z1
        + aux_cap_helper__b8,r4,band_LOW_PAIRED__condX1 <= 0
 c3192: - aux_cap_helper__b0,r11,band_HIGH_PAIRED__condX1 <= 0
 c3193: - aux_c_b5,r0 + sv_function_helpervar_b5,r0__condX1
        + sv_function_helpervar_b5,r0__condX3
        + sv_function_helpervar_b5,r0__condX2  = 0
 c3194: - aux_cap_helper__b7,r0,band_UNPAIRED__condX1 <= 0
 c3195: aux_cap_helper__b7,r2,band_LOW_PAIRED__condX1
        - aux_cap_helper__b7,r2,band_LOW_PAIRED__Z1 <= 0
 c3196: sv_function_helpervar_b9,r13__condX3
        - 15.9946187432626 sv_function_helpervar_b9,r13__Z3 <= 0
 c3197: aux_cap_helper__b1,r7,band_LOW_PAIRED__Z1
        + aux_cap_helper__b1,r7,band_LOW_PAIRED__Z2  = 1
 c3198: aux_cap_helper__b7,r2,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b7,r2,band_LOW_PAIRED__Z2 <= 0
 c3199: aux_cap_b5,r7,band_UNPAIRED - aux_c_b5,r7
        + aux_cap_b5,r7,band_HIGH_PAIRED + aux_cap_b5,r7,band_LOW_PAIRED  = 0
 c3200: aux_cap_helper__b1,r8,band_UNPAIRED__condX1
        + aux_cap_helper__b1,r8,band_UNPAIRED__condX2 - X__b1,r8,band_UNPAIRED
         = 0
 c3201: sv_function_helpervar_b9,r11__condX1
        + sv_function_helpervar_b9,r11__condX3
        + sv_function_helpervar_b9,r11__condX2 - aux_c_b9,r11  = 0
 c3202: aux_cap_helper__b0,r5,band_LOW_PAIRED__condX1
        + aux_cap_helper__b0,r5,band_LOW_PAIRED__condX2
        - X__b0,r5,band_LOW_PAIRED  = 0
 c3203: aux_cap_helper__b6,r10,band_UNPAIRED__condX2
        + aux_cap_helper__b6,r10,band_UNPAIRED__condX1
        - X__b6,r10,band_UNPAIRED  = 0
 c3204: aux_cap_helper__b9,r6,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b9,r6,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b9,r6,band_HIGH_PAIRED__condX2
        - X__b9,r6,band_HIGH_PAIRED  = 0
 c3205: aux_cap_helper__b6,r6,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b6,r6,band_HIGH_PAIRED__Z1 <= 0
 c3206: 0.0102040816326531 X__b8,r10,band_HIGH_PAIRED
        + 0.0102040816326531 X__b8,r10,band_LOW_PAIRED
        - GlobalBidder_W_ir__b8,r10
        + 0.0102040816326531 X__b8,r10,band_UNPAIRED <= 0
 c3207: - aux_cap_helper__b0,r9,band_LOW_PAIRED__condX1 <= 0
 c3208: aux_cap_helper__b9,r3,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b9,r3,band_HIGH_PAIRED__Z1 <= 0
 c3209: sv_function_helpervar_b9,r6__condX3
        - 15.9946187432626 sv_function_helpervar_b9,r6__Z3 <= 0
 c3210: - sv_function_helpervar_b0,r2__condX3
        + 6.44480188964475 sv_function_helpervar_b0,r2__Z3 <= 0
 c3211: aux_cap_helper__b8,r8,band_LOW_PAIRED__Z2
        - aux_cap_helper__b8,r8,band_LOW_PAIRED__condX2 <= 0
 c3212: - aux_cap_helper__b5,r4,band_HIGH_PAIRED__condX1 <= 0
 c3213: - aux_cap_helper__b7,r10,band_UNPAIRED__condX1 <= 0
 c3214: - sv_function_helpervar_b8,r12__condX1 <= 0
 c3215: - aux_Omega_b1,r8 + 273.810403119803 aux_sv_b1,r8  = 0
 c3216: - aux_cap_helper__b1,r5,band_HIGH_PAIRED__condX1 <= 0
 c3217: - aux_cap_b1,r3,band_LOW_PAIRED
        + 5.2303475434 aux_cap_helper__b1,r3,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b1,r3,band_LOW_PAIRED__condX1
        - 1.4943850124 aux_cap_helper__b1,r3,band_LOW_PAIRED__Z2  = 0
 c3218: - aux_cap_helper__b1,r1,band_UNPAIRED__condX1 <= 0
 c3219: - aux_cap_helper__b8,r2,band_UNPAIRED__condX1 <= 0
 c3220: - aux_cap_helper__b8,r1,band_HIGH_PAIRED__condX1 <= 0
 c3221: sv_function_helpervar_b2,r13__Z2 + sv_function_helpervar_b2,r13__Z1
        + sv_function_helpervar_b2,r13__Z3  = 1
 c3222: - sv_function_helpervar_b2,r1__condX2
        + 0.159946187435238 sv_function_helpervar_b2,r1__Z2 <= 0
 c3223: - 0.3214355447 aux_cap_helper__b7,r5,band_UNPAIRED__Z2
        - aux_cap_b7,r5,band_UNPAIRED
        + 0.8035888618 aux_cap_helper__b7,r5,band_UNPAIRED__condX1
        + 1.1250244065 aux_cap_helper__b7,r5,band_UNPAIRED__condX2  = 0
 c3224: - aux_cap_b1,r1,band_UNPAIRED
        - 0.3214355447 aux_cap_helper__b1,r1,band_UNPAIRED__Z2
        + 0.8035888618 aux_cap_helper__b1,r1,band_UNPAIRED__condX1
        + 1.1250244065 aux_cap_helper__b1,r1,band_UNPAIRED__condX2  = 0
 c3225: aux_cap_helper__b3,r0,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b3,r0,band_HIGH_PAIRED__condX2 <= 0
 c3226: sv_function_helpervar_b6,r6__Z2 + sv_function_helpervar_b6,r6__Z1
        + sv_function_helpervar_b6,r6__Z3  = 1
 c3227: 0.0102040816326531 X__b9,r3,band_LOW_PAIRED
        + 0.0102040816326531 X__b9,r3,band_UNPAIRED - GlobalBidder_W_ir__b9,r3
        + 0.0102040816326531 X__b9,r3,band_HIGH_PAIRED <= 0
 c3228: - aux_Omega_b9,r0 + 249.826049879819 aux_sv_b9,r0  = 0
 c3229: - 5.89346615134898 sv_function_helpervar_b2,r9__Z2
        + sv_function_helpervar_b2,r9__condX2 <= 0
 c3230: - aux_sv_b9,r11 + 1719.0945874076 sv_function_helpervar_b9,r11__condX1
        + 33.4147735735 sv_function_helpervar_b9,r11__condX3
        + 61.5909940435 sv_function_helpervar_b9,r11__condX2
        + 265.111380550982 sv_function_helpervar_b9,r11__Z2
        + 483.92352975758 sv_function_helpervar_b9,r11__Z3  = 0
 c3231: aux_cap_helper__b4,r8,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b4,r8,band_LOW_PAIRED__Z2 <= 0
 c3232: - aux_cap_b3,r7,band_LOW_PAIRED
        - 1.4943850124 aux_cap_helper__b3,r7,band_LOW_PAIRED__Z2
        + 5.2303475434 aux_cap_helper__b3,r7,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b3,r7,band_LOW_PAIRED__condX1  = 0
 c3233: 13 GlobalBidder_hat{W}__i8,k2 + GlobalBidder_W_i_8 >= 13
 c3234: - 0.159946187436967 sv_function_helpervar_b8,r9__Z1
        + sv_function_helpervar_b8,r9__condX1 <= 0
 c3235: - aux_cap_helper__b3,r3,band_LOW_PAIRED__Z1
        + aux_cap_helper__b3,r3,band_LOW_PAIRED__condX1 <= 0
 c3236: sv_function_helpervar_b0,r9__Z2 + sv_function_helpervar_b0,r9__Z3
        + sv_function_helpervar_b0,r9__Z1  = 1
 c3237: - aux_cap_b9,r8,band_LOW_PAIRED
        - 1.4943850124 aux_cap_helper__b9,r8,band_LOW_PAIRED__Z2
        + 5.2303475434 aux_cap_helper__b9,r8,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b9,r8,band_LOW_PAIRED__condX1  = 0
 c3238: - aux_Omega_b3,r4 + 1051.38876535178 aux_sv_b3,r4  = 0
 c3239: - sv_function_helpervar_b2,r4__condX2
        + 0.159946187439258 sv_function_helpervar_b2,r4__Z2 <= 0
 c3240: - aux_cap_helper__b2,r6,band_HIGH_PAIRED__condX1 <= 0
 c3241: aux_cap_helper__b9,r2,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b9,r2,band_HIGH_PAIRED__condX2 <= 0
 c3242: - 15.9946187432626 sv_function_helpervar_b2,r9__Z3
        + sv_function_helpervar_b2,r9__condX3 <= 0
 c3243: - 2 aux_cap_helper__b5,r3,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b5,r3,band_HIGH_PAIRED__condX2 <= 0
 c3244: aux_cap_helper__b4,r7,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b4,r7,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b4,r7,band_HIGH_PAIRED__condX2
        - X__b4,r7,band_HIGH_PAIRED  = 0
 c3245: - aux_Omega_b7,r13 + 40.9586980983049 aux_sv_b7,r13  = 0
 c3246: aux_cap_b5,r4,band_LOW_PAIRED - aux_c_b5,r4
        + aux_cap_b5,r4,band_UNPAIRED + aux_cap_b5,r4,band_HIGH_PAIRED  = 0
 c3247: - 15.9946187432626 sv_function_helpervar_b5,r3__Z3
        + sv_function_helpervar_b5,r3__condX3 <= 0
 c3248: - 1.4943850124 aux_cap_helper__b5,r1,band_LOW_PAIRED__Z2
        - aux_cap_b5,r1,band_LOW_PAIRED
        + 5.2303475434 aux_cap_helper__b5,r1,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b5,r1,band_LOW_PAIRED__condX1  = 0
 c3249: - 0.159946187435134 sv_function_helpervar_b7,r10__Z1
        + sv_function_helpervar_b7,r10__condX1 <= 0
 c3250: 0.159946187462509 sv_function_helpervar_b8,r3__Z2
        - sv_function_helpervar_b8,r3__condX2 <= 0
 c3251: 6.22135306242807 sv_function_helpervar_b3,r5__Z3
        - sv_function_helpervar_b3,r5__condX3 <= 0
 c3252: - 2 aux_cap_helper__b6,r10,band_LOW_PAIRED__Z2
        + aux_cap_helper__b6,r10,band_LOW_PAIRED__condX2 <= 0
 c3253: aux_cap_helper__b4,r0,band_UNPAIRED__condX1
        - aux_cap_helper__b4,r0,band_UNPAIRED__Z1 <= 0
 c3254: aux_cap_helper__b6,r1,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b6,r1,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b6,r1,band_HIGH_PAIRED__Z3  = 1
 c3255: - 5.76668212901648 sv_function_helpervar_b1,r1__Z2
        + sv_function_helpervar_b1,r1__condX2 <= 0
 c3256: aux_cap_helper__b4,r1,band_UNPAIRED__Z2
        + aux_cap_helper__b4,r1,band_UNPAIRED__Z1  = 1
 c3257: aux_cap_helper__b7,r5,band_LOW_PAIRED__condX1
        - aux_cap_helper__b7,r5,band_LOW_PAIRED__Z1 <= 0
 c3258: aux_cap_helper__b8,r6,band_UNPAIRED__Z1
        + aux_cap_helper__b8,r6,band_UNPAIRED__Z2  = 1
 c3259: - aux_cap_helper__b9,r11,band_LOW_PAIRED__condX2
        + aux_cap_helper__b9,r11,band_LOW_PAIRED__Z2 <= 0
 c3260: sv_function_helpervar_b1,r12__condX2
        - 5.6540979847395 sv_function_helpervar_b1,r12__Z2 <= 0
 c3261: - aux_cap_helper__b9,r10,band_UNPAIRED__condX1 <= 0
 c3262: sv_function_helpervar_b7,r12__condX3
        - 15.9946187432626 sv_function_helpervar_b7,r12__Z3 <= 0
 c3263: aux_cap_helper__b0,r10,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b0,r10,band_UNPAIRED__Z2 <= 0
 c3264: - aux_cap_helper__b9,r5,band_UNPAIRED__condX2
        + aux_cap_helper__b9,r5,band_UNPAIRED__Z2 <= 0
 c3265: - aux_cap_helper__b9,r12,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b9,r12,band_HIGH_PAIRED__Z3 <= 0
 c3266: aux_cap_helper__b4,r9,band_LOW_PAIRED__condX1
        - aux_cap_helper__b4,r9,band_LOW_PAIRED__Z1 <= 0
 c3267: sv_function_helpervar_b3,r8__condX1
        - 0.159946187466617 sv_function_helpervar_b3,r8__Z1 <= 0
 c3268: 0.0102040816326531 X__b8,r13,band_UNPAIRED
        + 0.0102040816326531 X__b8,r13,band_LOW_PAIRED
        - GlobalBidder_W_ir__b8,r13
        + 0.0102040816326531 X__b8,r13,band_HIGH_PAIRED <= 0
 c3269: 440.250487366622 sv_function_helpervar_b4,r6__Z3
        + 196.258664312973 sv_function_helpervar_b4,r6__Z2
        + 20.2279487395 sv_function_helpervar_b4,r6__condX3
        + 1289.3272633345 sv_function_helpervar_b4,r6__condX1
        + 62.2979266118 sv_function_helpervar_b4,r6__condX2 - aux_sv_b4,r6  = 0
 c3270: - aux_cap_helper__b6,r7,band_LOW_PAIRED__condX1 <= 0
 c3271: - aux_cap_helper__b5,r3,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b5,r3,band_HIGH_PAIRED__condX1 <= 0
 c3272: - X__b4,r2,band_LOW_PAIRED
        + aux_cap_helper__b4,r2,band_LOW_PAIRED__condX2
        + aux_cap_helper__b4,r2,band_LOW_PAIRED__condX1  = 0
 c3273: - aux_cap_b2,r9,band_UNPAIRED
        - 0.3214355447 aux_cap_helper__b2,r9,band_UNPAIRED__Z2
        + 1.1250244065 aux_cap_helper__b2,r9,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b2,r9,band_UNPAIRED__condX1  = 0
 c3274: sv_function_helpervar_b6,r2__Z3 + sv_function_helpervar_b6,r2__Z1
        + sv_function_helpervar_b6,r2__Z2  = 1
 c3275: aux_cap_helper__b4,r9,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b4,r9,band_LOW_PAIRED__Z2 <= 0
 c3276: aux_cap_helper__b3,r4,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b3,r4,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b3,r4,band_HIGH_PAIRED__condX3
        - X__b3,r4,band_HIGH_PAIRED  = 0
 c3277: aux_cap_helper__b7,r8,band_UNPAIRED__condX1
        - aux_cap_helper__b7,r8,band_UNPAIRED__Z1 <= 0
 c3278: aux_cap_helper__b5,r11,band_LOW_PAIRED__Z1
        + aux_cap_helper__b5,r11,band_LOW_PAIRED__Z2  = 1
 c3279: - aux_cap_helper__b6,r10,band_LOW_PAIRED__Z1
        + aux_cap_helper__b6,r10,band_LOW_PAIRED__condX1 <= 0
 c3280: - X__b1,r8,band_LOW_PAIRED
        + aux_cap_helper__b1,r8,band_LOW_PAIRED__condX1
        + aux_cap_helper__b1,r8,band_LOW_PAIRED__condX2  = 0
 c3281: - 2 aux_cap_helper__b3,r3,band_LOW_PAIRED__Z2
        + aux_cap_helper__b3,r3,band_LOW_PAIRED__condX2 <= 0
 c3282: - 0.6181448971 aux_cap_helper__b8,r13,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b8,r13,band_HIGH_PAIRED__Z3
        + 1.5453622427 aux_cap_helper__b8,r13,band_HIGH_PAIRED__condX1
        + 1.3908260184 aux_cap_helper__b8,r13,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b8,r13,band_HIGH_PAIRED__condX2
        - aux_cap_b8,r13,band_HIGH_PAIRED  = 0
 c3283: - X__b2,r7,band_UNPAIRED + aux_cap_helper__b2,r7,band_UNPAIRED__condX1
        + aux_cap_helper__b2,r7,band_UNPAIRED__condX2  = 0
 c3284: aux_cap_helper__b5,r5,band_LOW_PAIRED__Z1
        + aux_cap_helper__b5,r5,band_LOW_PAIRED__Z2  = 1
 c3285: aux_cap_b7,r0,band_UNPAIRED - aux_c_b7,r0
        + aux_cap_b7,r0,band_LOW_PAIRED + aux_cap_b7,r0,band_HIGH_PAIRED  = 0
 c3286: sv_function_helpervar_b0,r5__Z1 + sv_function_helpervar_b0,r5__Z2
        + sv_function_helpervar_b0,r5__Z3  = 1
 c3287: - 15.9946187432626 sv_function_helpervar_b1,r10__Z3
        + sv_function_helpervar_b1,r10__condX3 <= 0
 c3288: aux_cap_helper__b4,r8,band_LOW_PAIRED__condX1
        - aux_cap_helper__b4,r8,band_LOW_PAIRED__Z1 <= 0
 c3289: - 2 aux_cap_helper__b2,r1,band_UNPAIRED__Z2
        + aux_cap_helper__b2,r1,band_UNPAIRED__condX2 <= 0
 c3290: - aux_cap_helper__b2,r11,band_UNPAIRED__condX1 <= 0
 c3291: - sv_function_helpervar_b8,r7__condX3
        + 8.24676410539167 sv_function_helpervar_b8,r7__Z3 <= 0
 c3292: - 15.9946187432626 sv_function_helpervar_b0,r13__Z3
        + sv_function_helpervar_b0,r13__condX3 <= 0
 c3293: aux_cap_helper__b0,r10,band_UNPAIRED__condX1
        - aux_cap_helper__b0,r10,band_UNPAIRED__Z1 <= 0
 c3294: - aux_cap_b1,r5,band_UNPAIRED
        - 0.3214355447 aux_cap_helper__b1,r5,band_UNPAIRED__Z2
        + 1.1250244065 aux_cap_helper__b1,r5,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b1,r5,band_UNPAIRED__condX1  = 0
 c3295: - sv_function_helpervar_b0,r4__condX1 <= 0
 c3296: - X__b3,r0,band_UNPAIRED + aux_cap_helper__b3,r0,band_UNPAIRED__condX1
        + aux_cap_helper__b3,r0,band_UNPAIRED__condX2  = 0
 c3297: aux_cap_helper__b3,r10,band_LOW_PAIRED__condX1
        - aux_cap_helper__b3,r10,band_LOW_PAIRED__Z1 <= 0
 c3298: - 0.159946187453678 sv_function_helpervar_b4,r11__Z1
        + sv_function_helpervar_b4,r11__condX1 <= 0
 c3299: - X__b6,r1,band_UNPAIRED + aux_cap_helper__b6,r1,band_UNPAIRED__condX1
        + aux_cap_helper__b6,r1,band_UNPAIRED__condX2  = 0
 c3300: - aux_cap_helper__b9,r12,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b9,r12,band_HIGH_PAIRED__Z2 <= 0
 c3301: - aux_cap_helper__b3,r2,band_LOW_PAIRED__condX1 <= 0
 c3302: aux_cap_helper__b7,r9,band_LOW_PAIRED__Z2
        - aux_cap_helper__b7,r9,band_LOW_PAIRED__condX2 <= 0
 c3303: - 2 aux_cap_helper__b9,r7,band_UNPAIRED__Z2
        + aux_cap_helper__b9,r7,band_UNPAIRED__condX2 <= 0
 c3304: aux_cap_helper__b0,r10,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b0,r10,band_HIGH_PAIRED__Z2 <= 0
 c3305: sv_function_helpervar_b8,r3__Z1 + sv_function_helpervar_b8,r3__Z3
        + sv_function_helpervar_b8,r3__Z2  = 1
 c3306: - aux_cap_helper__b0,r2,band_LOW_PAIRED__condX1 <= 0
 c3307: aux_cap_helper__b8,r2,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b8,r2,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b8,r2,band_HIGH_PAIRED__Z3  = 1
 c3308: aux_cap_helper__b6,r2,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b6,r2,band_HIGH_PAIRED__Z3 <= 0
 c3309: aux_cap_helper__b3,r10,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b3,r10,band_LOW_PAIRED__Z2 <= 0
 c3310: aux_cap_helper__b8,r3,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b8,r3,band_HIGH_PAIRED__Z1 <= 0
 c3311: - X__b8,r9,band_HIGH_PAIRED
        + aux_cap_helper__b8,r9,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b8,r9,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b8,r9,band_HIGH_PAIRED__condX2  = 0
 c3312: - aux_Omega_b7,r9 + 427.619720431404 aux_sv_b7,r9  = 0
 c3313: - aux_cap_helper__b9,r4,band_LOW_PAIRED__condX1 <= 0
 c3314: aux_cap_helper__b4,r8,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b4,r8,band_UNPAIRED__Z2 <= 0
 c3315: - aux_cap_helper__b6,r8,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b6,r8,band_HIGH_PAIRED__condX1 <= 0
 c3316: aux_cap_helper__b2,r13,band_UNPAIRED__condX1
        + aux_cap_helper__b2,r13,band_UNPAIRED__condX2
        - X__b2,r13,band_UNPAIRED  = 0
 c3317: 0.927217345653617 aux_cap_helper__b5,r10,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b5,r10,band_HIGH_PAIRED__Z2
        - aux_cap_b5,r10,band_HIGH_PAIRED
        + 1.5453622427 aux_cap_helper__b5,r10,band_HIGH_PAIRED__condX1
        + 1.3908260184 aux_cap_helper__b5,r10,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b5,r10,band_HIGH_PAIRED__condX2  = 0
 c3318: aux_cap_helper__b6,r5,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b6,r5,band_HIGH_PAIRED__condX2 <= 0
 c3319: aux_cap_helper__b0,r13,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b0,r13,band_HIGH_PAIRED__Z1 <= 0
 c3320: aux_cap_helper__b8,r3,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b8,r3,band_HIGH_PAIRED__Z2 <= 0
 c3321: aux_cap_helper__b4,r6,band_LOW_PAIRED__Z2
        + aux_cap_helper__b4,r6,band_LOW_PAIRED__Z1  = 1
 c3322: - sv_function_helpervar_b6,r11__condX3
        + 6.30329523784073 sv_function_helpervar_b6,r11__Z3 <= 0
 c3323: - aux_cap_helper__b9,r0,band_LOW_PAIRED__condX1 <= 0
 c3324: - aux_cap_helper__b2,r1,band_UNPAIRED__Z1
        + aux_cap_helper__b2,r1,band_UNPAIRED__condX1 <= 0
 c3325: - aux_cap_helper__b8,r12,band_LOW_PAIRED__condX2
        + aux_cap_helper__b8,r12,band_LOW_PAIRED__Z2 <= 0
 c3326: sv_function_helpervar_b1,r10__condX1
        + sv_function_helpervar_b1,r10__condX2
        + sv_function_helpervar_b1,r10__condX3 - aux_c_b1,r10  = 0
 c3327: aux_cap_helper__b4,r9,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b4,r9,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b4,r9,band_HIGH_PAIRED__condX1
        - X__b4,r9,band_HIGH_PAIRED  = 0
 c3328: aux_cap_helper__b0,r8,band_LOW_PAIRED__Z1
        + aux_cap_helper__b0,r8,band_LOW_PAIRED__Z2  = 1
 c3329: aux_cap_helper__b7,r8,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b7,r8,band_UNPAIRED__Z2 <= 0
 c3330: aux_cap_helper__b0,r10,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b0,r10,band_HIGH_PAIRED__Z1 <= 0
 c3331: aux_cap_helper__b8,r0,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b8,r0,band_HIGH_PAIRED__Z2 <= 0
 c3332: - aux_cap_helper__b6,r2,band_UNPAIRED__condX1 <= 0
 c3333: - sv_function_helpervar_b6,r6__condX3
        + 5.80305813698008 sv_function_helpervar_b6,r6__Z3 <= 0
 c3334: - 5.74126635300054 sv_function_helpervar_b3,r1__Z2
        + sv_function_helpervar_b3,r1__condX2 <= 0
 c3335: - sv_function_helpervar_b7,r1__condX1 <= 0
 c3336: aux_cap_helper__b0,r10,band_UNPAIRED__condX2
        + aux_cap_helper__b0,r10,band_UNPAIRED__condX1
        - X__b0,r10,band_UNPAIRED  = 0
 c3337: - aux_c_b7,r5 + aux_cap_b7,r5,band_HIGH_PAIRED
        + aux_cap_b7,r5,band_UNPAIRED + aux_cap_b7,r5,band_LOW_PAIRED  = 0
 c3338: aux_cap_helper__b6,r2,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b6,r2,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b6,r2,band_HIGH_PAIRED__condX2
        - X__b6,r2,band_HIGH_PAIRED  = 0
 c3339: aux_cap_helper__b1,r2,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b1,r2,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b1,r2,band_HIGH_PAIRED__Z3  = 1
 c3340: - aux_cap_helper__b2,r9,band_HIGH_PAIRED__condX1 <= 0
 c3341: aux_cap_helper__b9,r5,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b9,r5,band_HIGH_PAIRED__condX2 <= 0
 c3342: aux_cap_helper__b3,r2,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b3,r2,band_HIGH_PAIRED__Z3 <= 0
 c3343: - 2 aux_cap_helper__b8,r7,band_LOW_PAIRED__Z2
        + aux_cap_helper__b8,r7,band_LOW_PAIRED__condX2 <= 0
 c3344: - aux_cap_helper__b0,r5,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b0,r5,band_HIGH_PAIRED__Z3 <= 0
 c3345: 0.0102040816326531 X__b7,r12,band_HIGH_PAIRED
        + 0.0102040816326531 X__b7,r12,band_UNPAIRED
        + 0.0102040816326531 X__b7,r12,band_LOW_PAIRED
        - GlobalBidder_W_ir__b7,r12 <= 0
 c3346: sv_function_helpervar_b8,r6__condX1
        - 0.159946187438625 sv_function_helpervar_b8,r6__Z1 <= 0
 c3347: - aux_cap_helper__b3,r12,band_HIGH_PAIRED__condX1 <= 0
 c3348: aux_cap_helper__b7,r5,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b7,r5,band_LOW_PAIRED__Z2 <= 0
 c3349: - aux_cap_helper__b8,r11,band_HIGH_PAIRED__condX1 <= 0
 c3350: aux_cap_helper__b2,r3,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b2,r3,band_HIGH_PAIRED__Z3 <= 0
 c3351: aux_cap_b9,r1,band_HIGH_PAIRED + aux_cap_b9,r1,band_LOW_PAIRED
        + aux_cap_b9,r1,band_UNPAIRED - aux_c_b9,r1  = 0
 c3352: - aux_cap_helper__b0,r6,band_LOW_PAIRED__condX1 <= 0
 c3353: 408.273979178829 sv_function_helpervar_b5,r7__Z3
        + 177.491627515393 sv_function_helpervar_b5,r7__Z2 - aux_sv_b5,r7
        + 1169.597442413 sv_function_helpervar_b5,r7__condX1
        + 59.901548059 sv_function_helpervar_b5,r7__condX2
        + 17.7927150709 sv_function_helpervar_b5,r7__condX3  = 0
 c3354: 2 aux_cap_helper__b9,r5,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b9,r5,band_HIGH_PAIRED__condX3 <= 0
 c3355: - aux_cap_helper__b5,r6,band_HIGH_PAIRED__condX1 <= 0
 c3356: - aux_Omega_b7,r4 + 1865.85119296766 aux_sv_b7,r4  = 0
 c3357: 1.1250244065 aux_cap_helper__b9,r5,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b9,r5,band_UNPAIRED__condX1
        - 0.3214355447 aux_cap_helper__b9,r5,band_UNPAIRED__Z2
        - aux_cap_b9,r5,band_UNPAIRED  = 0
 c3358: - 1.4943850124 aux_cap_helper__b8,r2,band_LOW_PAIRED__Z2
        - aux_cap_b8,r2,band_LOW_PAIRED
        + 5.2303475434 aux_cap_helper__b8,r2,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b8,r2,band_LOW_PAIRED__condX1  = 0
 c3359: sv_function_helpervar_b8,r12__condX1
        - 0.159946187470694 sv_function_helpervar_b8,r12__Z1 <= 0
 c3360: sv_function_helpervar_b1,r11__condX1
        + sv_function_helpervar_b1,r11__condX2
        + sv_function_helpervar_b1,r11__condX3 - aux_c_b1,r11  = 0
 c3361: sv_function_helpervar_b2,r3__condX2
        - 6.24106605787985 sv_function_helpervar_b2,r3__Z2 <= 0
 c3362: sv_function_helpervar_b0,r12__condX1
        - 0.159946187446023 sv_function_helpervar_b0,r12__Z1 <= 0
 c3363: - sv_function_helpervar_b1,r10__condX1 <= 0
 c3364: 0.159946187434125 sv_function_helpervar_b1,r10__Z2
        - sv_function_helpervar_b1,r10__condX2 <= 0
 c3365: - 2 aux_cap_helper__b3,r8,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b3,r8,band_HIGH_PAIRED__condX2 <= 0
 c3366: - 15.9946187432626 sv_function_helpervar_b0,r4__Z3
        + sv_function_helpervar_b0,r4__condX3 <= 0
 c3367: - 2 aux_cap_helper__b6,r8,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b6,r8,band_HIGH_PAIRED__condX2 <= 0
 c3368: aux_cap_helper__b9,r3,band_LOW_PAIRED__Z1
        + aux_cap_helper__b9,r3,band_LOW_PAIRED__Z2  = 1
 c3369: aux_cap_helper__b3,r0,band_UNPAIRED__Z2
        + aux_cap_helper__b3,r0,band_UNPAIRED__Z1  = 1
 c3370: - X__b6,r13,band_HIGH_PAIRED
        + aux_cap_helper__b6,r13,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b6,r13,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b6,r13,band_HIGH_PAIRED__condX1  = 0
 c3371: aux_cap_helper__b0,r13,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b0,r13,band_HIGH_PAIRED__Z2 <= 0
 c3372: 2 aux_cap_helper__b6,r5,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b6,r5,band_HIGH_PAIRED__condX3 <= 0
 c3373: aux_cap_helper__b2,r2,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b2,r2,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b2,r2,band_HIGH_PAIRED__Z2  = 1
 c3374: - 3 aux_cap_helper__b2,r13,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b2,r13,band_HIGH_PAIRED__condX3 <= 0
 c3375: aux_cap_helper__b7,r12,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b7,r12,band_HIGH_PAIRED__Z3 <= 0
 c3376: - aux_cap_helper__b9,r7,band_UNPAIRED__Z1
        + aux_cap_helper__b9,r7,band_UNPAIRED__condX1 <= 0
 c3377: 2 aux_cap_helper__b9,r2,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b9,r2,band_HIGH_PAIRED__condX3 <= 0
 c3378: - aux_cap_helper__b8,r7,band_LOW_PAIRED__Z1
        + aux_cap_helper__b8,r7,band_LOW_PAIRED__condX1 <= 0
 c3379: - aux_Omega_b1,r5 + 380.963137702469 aux_sv_b1,r5  = 0
 c3380: - 0.15994618744377 sv_function_helpervar_b4,r12__Z1
        + sv_function_helpervar_b4,r12__condX1 <= 0
 c3381: - aux_cap_helper__b3,r10,band_UNPAIRED__condX2
        + aux_cap_helper__b3,r10,band_UNPAIRED__Z2 <= 0
 c3382: 594.680999510566 aux_sv_b4,r10 - aux_Omega_b4,r10  = 0
 c3383: - aux_cap_b4,r3,band_UNPAIRED
        - 0.3214355447 aux_cap_helper__b4,r3,band_UNPAIRED__Z2
        + 0.8035888618 aux_cap_helper__b4,r3,band_UNPAIRED__condX1
        + 1.1250244065 aux_cap_helper__b4,r3,band_UNPAIRED__condX2  = 0
 c3384: X__b8,r8,band_HIGH_PAIRED + X__b8,r8,band_LOW_PAIRED
        - GlobalBidder_W_ir__b8,r8 + X__b8,r8,band_UNPAIRED >= 0
 c3385: sv_function_helpervar_b6,r8__condX3
        + sv_function_helpervar_b6,r8__condX1
        + sv_function_helpervar_b6,r8__condX2 - aux_c_b6,r8  = 0
 c3386: aux_cap_helper__b9,r5,band_LOW_PAIRED__Z2
        + aux_cap_helper__b9,r5,band_LOW_PAIRED__Z1  = 1
 c3387: - aux_cap_helper__b2,r10,band_UNPAIRED__condX1 <= 0
 c3388: - aux_cap_helper__b1,r8,band_HIGH_PAIRED__condX1 <= 0
 c3389: - 0.3214355447 aux_cap_helper__b5,r12,band_UNPAIRED__Z2
        + 1.1250244065 aux_cap_helper__b5,r12,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b5,r12,band_UNPAIRED__condX1
        - aux_cap_b5,r12,band_UNPAIRED  = 0
 c3390: - aux_cap_helper__b0,r11,band_LOW_PAIRED__condX1 <= 0
 c3391: - aux_cap_helper__b9,r7,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b9,r7,band_HIGH_PAIRED__Z2 <= 0
 c3392: X__b7,r13,band_HIGH_PAIRED + X__b4,r13,band_HIGH_PAIRED
        + X__b2,r13,band_HIGH_PAIRED + X__b6,r13,band_HIGH_PAIRED
        + X__b0,r13,band_HIGH_PAIRED + X__b1,r13,band_HIGH_PAIRED
        + X__b3,r13,band_HIGH_PAIRED + X__b5,r13,band_HIGH_PAIRED
        + X__b8,r13,band_HIGH_PAIRED + X__b9,r13,band_HIGH_PAIRED <= 3
 c3393: sv_function_helpervar_b8,r7__condX1
        + sv_function_helpervar_b8,r7__condX3
        + sv_function_helpervar_b8,r7__condX2 - aux_c_b8,r7  = 0
 c3394: X__b5,r6,band_UNPAIRED + X__b4,r6,band_UNPAIRED
        + X__b6,r6,band_UNPAIRED + X__b2,r6,band_UNPAIRED
        + X__b8,r6,band_UNPAIRED + X__b3,r6,band_UNPAIRED
        + X__b7,r6,band_UNPAIRED + X__b1,r6,band_UNPAIRED
        + X__b9,r6,band_UNPAIRED + X__b0,r6,band_UNPAIRED <= 2
 c3395: aux_cap_helper__b1,r9,band_LOW_PAIRED__condX1
        - aux_cap_helper__b1,r9,band_LOW_PAIRED__Z1 <= 0
 c3396: - X__b2,r8,band_HIGH_PAIRED
        + aux_cap_helper__b2,r8,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b2,r8,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b2,r8,band_HIGH_PAIRED__condX3  = 0
 c3397: - X__b6,r3,band_LOW_PAIRED
        + aux_cap_helper__b6,r3,band_LOW_PAIRED__condX1
        + aux_cap_helper__b6,r3,band_LOW_PAIRED__condX2  = 0
 c3398: - aux_cap_helper__b7,r8,band_LOW_PAIRED__Z1
        + aux_cap_helper__b7,r8,band_LOW_PAIRED__condX1 <= 0
 c3399: - aux_Omega_b6,r9 + 320.102147901441 aux_sv_b6,r9  = 0
 c3400: aux_cap_helper__b7,r10,band_UNPAIRED__condX2
        + aux_cap_helper__b7,r10,band_UNPAIRED__condX1
        - X__b7,r10,band_UNPAIRED  = 0
 c3401: sv_function_helpervar_b9,r4__Z1 + sv_function_helpervar_b9,r4__Z2
        + sv_function_helpervar_b9,r4__Z3  = 1
 c3402: aux_cap_helper__b7,r1,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b7,r1,band_LOW_PAIRED__Z2 <= 0
 c3403: - aux_cap_helper__b6,r13,band_LOW_PAIRED__Z1
        + aux_cap_helper__b6,r13,band_LOW_PAIRED__condX1 <= 0
 c3404: sv_function_helpervar_b4,r9__condX2
        - 5.68428432618719 sv_function_helpervar_b4,r9__Z2 <= 0
 c3405: - 15.9946187432626 sv_function_helpervar_b6,r0__Z3
        + sv_function_helpervar_b6,r0__condX3 <= 0
 c3406: - aux_cap_helper__b9,r7,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b9,r7,band_HIGH_PAIRED__Z3 <= 0
 c3407: - 2 aux_cap_helper__b7,r8,band_LOW_PAIRED__Z2
        + aux_cap_helper__b7,r8,band_LOW_PAIRED__condX2 <= 0
 c3408: 0.03125 aux_Omega_b6,r10 + 0.01314 aux_Omega_b6,r11
        + 0.00552 aux_Omega_b6,r12 + 0.01314 aux_Omega_b6,r13 - v__6
        + 0.42045 aux_Omega_b6,r3 + 0.17678 aux_Omega_b6,r4
        + 0.07433 aux_Omega_b6,r5 + 0.42045 aux_Omega_b6,r6
        + 0.07433 aux_Omega_b6,r7 + 0.17678 aux_Omega_b6,r8
        + 0.07433 aux_Omega_b6,r9 + aux_Omega_b6,r0 + 0.42045 aux_Omega_b6,r1
        + 0.17678 aux_Omega_b6,r2  = 0
 c3409: - 2 aux_cap_helper__b2,r11,band_LOW_PAIRED__Z2
        + aux_cap_helper__b2,r11,band_LOW_PAIRED__condX2 <= 0
 c3410: - aux_cap_helper__b3,r8,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b3,r8,band_HIGH_PAIRED__condX1 <= 0
 c3411: sv_function_helpervar_b9,r6__Z2 + sv_function_helpervar_b9,r6__Z3
        + sv_function_helpervar_b9,r6__Z1  = 1
 c3412: aux_cap_helper__b3,r0,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b3,r0,band_LOW_PAIRED__Z2 <= 0
 c3413: sv_function_helpervar_b4,r10__condX1
        - 0.159946187444592 sv_function_helpervar_b4,r10__Z1 <= 0
 c3414: - sv_function_helpervar_b2,r3__condX2
        + 0.159946187533302 sv_function_helpervar_b2,r3__Z2 <= 0
 c3415: aux_cap_helper__b6,r4,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b6,r4,band_HIGH_PAIRED__Z3 <= 0
 c3416: - 0.159946187432645 sv_function_helpervar_b8,r5__Z1
        + sv_function_helpervar_b8,r5__condX1 <= 0
 c3417: - 3 aux_cap_helper__b5,r1,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b5,r1,band_HIGH_PAIRED__condX3 <= 0
 c3418: - aux_cap_helper__b5,r12,band_LOW_PAIRED__condX2
        + aux_cap_helper__b5,r12,band_LOW_PAIRED__Z2 <= 0
 c3419: aux_cap_helper__b1,r8,band_LOW_PAIRED__Z2
        + aux_cap_helper__b1,r8,band_LOW_PAIRED__Z1  = 1
 c3420: aux_cap_helper__b4,r1,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b4,r1,band_UNPAIRED__Z2 <= 0
 c3421: - aux_cap_helper__b5,r6,band_UNPAIRED__condX2
        + aux_cap_helper__b5,r6,band_UNPAIRED__Z2 <= 0
 c3422: - aux_cap_helper__b7,r12,band_UNPAIRED__condX1 <= 0
 c3423: - 5.8731634253912 sv_function_helpervar_b6,r10__Z2
        + sv_function_helpervar_b6,r10__condX2 <= 0
 c3424: - 5.6205873774191 sv_function_helpervar_b6,r5__Z2
        + sv_function_helpervar_b6,r5__condX2 <= 0
 c3425: - aux_cap_helper__b1,r3,band_UNPAIRED__condX1 <= 0
 c3426: - sv_function_helpervar_b7,r2__condX1 <= 0
 c3427: - aux_cap_helper__b2,r8,band_UNPAIRED__Z1
        + aux_cap_helper__b2,r8,band_UNPAIRED__condX1 <= 0
 c3428: - sv_function_helpervar_b1,r1__condX1 <= 0
 c3429: sv_function_helpervar_b5,r1__Z3 + sv_function_helpervar_b5,r1__Z2
        + sv_function_helpervar_b5,r1__Z1  = 1
 c3430: sv_function_helpervar_b4,r2__condX2
        - 5.59831642723657 sv_function_helpervar_b4,r2__Z2 <= 0
 c3431: - 15.9946187432626 sv_function_helpervar_b5,r11__Z3
        + sv_function_helpervar_b5,r11__condX3 <= 0
 c3432: sv_function_helpervar_b3,r2__condX3
        + sv_function_helpervar_b3,r2__condX2
        + sv_function_helpervar_b3,r2__condX1 - aux_c_b3,r2  = 0
 c3433: aux_cap_helper__b0,r7,band_UNPAIRED__Z2
        - aux_cap_helper__b0,r7,band_UNPAIRED__condX2 <= 0
 c3434: - aux_cap_helper__b3,r2,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b3,r2,band_HIGH_PAIRED__Z2 <= 0
 c3435: - aux_cap_helper__b1,r12,band_LOW_PAIRED__Z1
        + aux_cap_helper__b1,r12,band_LOW_PAIRED__condX1 <= 0
 c3436: 2.1635071398 aux_cap_helper__b5,r12,band_HIGH_PAIRED__condX2
        + 1.3908260184 aux_cap_helper__b5,r12,band_HIGH_PAIRED__condX3
        + 1.5453622427 aux_cap_helper__b5,r12,band_HIGH_PAIRED__condX1
        + 0.927217345653617 aux_cap_helper__b5,r12,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b5,r12,band_HIGH_PAIRED__Z2
        - aux_cap_b5,r12,band_HIGH_PAIRED  = 0
 c3437: aux_cap_helper__b2,r5,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b2,r5,band_HIGH_PAIRED__Z1 <= 0
 c3438: X__b7,r11,band_UNPAIRED + X__b2,r11,band_UNPAIRED
        + X__b9,r11,band_UNPAIRED + X__b1,r11,band_UNPAIRED
        + X__b5,r11,band_UNPAIRED + X__b3,r11,band_UNPAIRED
        + X__b6,r11,band_UNPAIRED + X__b0,r11,band_UNPAIRED
        + X__b4,r11,band_UNPAIRED + X__b8,r11,band_UNPAIRED <= 2
 c3439: aux_cap_helper__b4,r13,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b4,r13,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b4,r13,band_HIGH_PAIRED__Z2  = 1
 c3440: aux_cap_b5,r13,band_LOW_PAIRED - aux_c_b5,r13
        + aux_cap_b5,r13,band_UNPAIRED + aux_cap_b5,r13,band_HIGH_PAIRED  = 0
 c3441: aux_cap_helper__b3,r0,band_LOW_PAIRED__condX1
        - aux_cap_helper__b3,r0,band_LOW_PAIRED__Z1 <= 0
 c3442: 0.8035888618 aux_cap_helper__b7,r1,band_UNPAIRED__condX1
        + 1.1250244065 aux_cap_helper__b7,r1,band_UNPAIRED__condX2
        - 0.3214355447 aux_cap_helper__b7,r1,band_UNPAIRED__Z2
        - aux_cap_b7,r1,band_UNPAIRED  = 0
 c3443: - aux_cap_helper__b7,r8,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b7,r8,band_HIGH_PAIRED__Z3 <= 0
 c3444: - sv_function_helpervar_b5,r12__condX1 <= 0
 c3445: aux_cap_helper__b1,r9,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b1,r9,band_LOW_PAIRED__Z2 <= 0
 c3446: aux_cap_helper__b2,r5,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b2,r5,band_HIGH_PAIRED__Z2 <= 0
 c3447: - 2 aux_cap_helper__b6,r13,band_LOW_PAIRED__Z2
        + aux_cap_helper__b6,r13,band_LOW_PAIRED__condX2 <= 0
 c3448: - aux_cap_helper__b5,r11,band_HIGH_PAIRED__condX1 <= 0
 c3449: sv_function_helpervar_b7,r5__condX1
        - 0.159946187454911 sv_function_helpervar_b7,r5__Z1 <= 0
 c3450: aux_cap_helper__b0,r7,band_UNPAIRED__Z1
        + aux_cap_helper__b0,r7,band_UNPAIRED__Z2  = 1
 c3451: aux_cap_helper__b7,r5,band_UNPAIRED__Z2
        + aux_cap_helper__b7,r5,band_UNPAIRED__Z1  = 1
 c3452: - 15.9946187432626 sv_function_helpervar_b1,r0__Z3
        + sv_function_helpervar_b1,r0__condX3 <= 0
 c3453: aux_cap_helper__b1,r3,band_UNPAIRED__Z1
        + aux_cap_helper__b1,r3,band_UNPAIRED__Z2  = 1
 c3454: aux_cap_helper__b9,r1,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b9,r1,band_HIGH_PAIRED__Z2 <= 0
 c3455: - aux_cap_helper__b5,r13,band_UNPAIRED__condX1 <= 0
 c3456: - aux_cap_helper__b6,r4,band_LOW_PAIRED__condX1 <= 0
 c3457: 1.5453622427 aux_cap_helper__b6,r0,band_HIGH_PAIRED__condX1
        + 1.3908260184 aux_cap_helper__b6,r0,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b6,r0,band_HIGH_PAIRED__condX2
        - aux_cap_b6,r0,band_HIGH_PAIRED
        - 0.6181448971 aux_cap_helper__b6,r0,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b6,r0,band_HIGH_PAIRED__Z3  = 0
 c3458: aux_Omega_b9,r1 + aux_Omega_b9,r0 + aux_Omega_b9,r3 + aux_Omega_b9,r2
        + aux_Omega_b9,r5 + aux_Omega_b9,r4 + aux_Omega_b9,r7 + aux_Omega_b9,r6
        + aux_Omega_b9,r9 + aux_Omega_b9,r8 - GlobalBidder_PSI__i9,k1
        + aux_Omega_b9,r13 + aux_Omega_b9,r10 + aux_Omega_b9,r11
        + aux_Omega_b9,r12 + 18497526.8940961 GlobalBidder_hat{W}__i9,k1
        <= 18497526.8940961
 c3459: - aux_c_b4,r1 + sv_function_helpervar_b4,r1__condX1
        + sv_function_helpervar_b4,r1__condX2
        + sv_function_helpervar_b4,r1__condX3  = 0
 c3460: 1849.77609073879 aux_sv_b9,r12 - aux_Omega_b9,r12  = 0
 c3461: aux_cap_helper__b9,r6,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b9,r6,band_HIGH_PAIRED__Z1 <= 0
 c3462: aux_cap_helper__b6,r9,band_UNPAIRED__Z2
        + aux_cap_helper__b6,r9,band_UNPAIRED__Z1  = 1
 c3463: aux_cap_helper__b4,r7,band_UNPAIRED__condX1
        + aux_cap_helper__b4,r7,band_UNPAIRED__condX2 - X__b4,r7,band_UNPAIRED
         = 0
 c3464: 2 aux_cap_helper__b9,r10,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b9,r10,band_HIGH_PAIRED__condX3 <= 0
 c3465: - aux_cap_helper__b1,r10,band_LOW_PAIRED__condX1 <= 0
 c3466: aux_cap_b1,r9,band_UNPAIRED + aux_cap_b1,r9,band_LOW_PAIRED
        + aux_cap_b1,r9,band_HIGH_PAIRED - aux_c_b1,r9  = 0
 c3467: aux_cap_b1,r3,band_UNPAIRED + aux_cap_b1,r3,band_HIGH_PAIRED
        + aux_cap_b1,r3,band_LOW_PAIRED - aux_c_b1,r3  = 0
 c3468: - aux_cap_helper__b7,r8,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b7,r8,band_HIGH_PAIRED__Z2 <= 0
 c3469: GlobalBidder_PSI__i7,k0 - 18497526.8940961 GlobalBidder_hat{W}__i7,k0
        <= 0
 c3470: sv_function_helpervar_b8,r10__Z1 + sv_function_helpervar_b8,r10__Z2
        + sv_function_helpervar_b8,r10__Z3  = 1
 c3471: sv_function_helpervar_b7,r13__condX1
        - 0.159946187553672 sv_function_helpervar_b7,r13__Z1 <= 0
 c3472: aux_cap_helper__b6,r11,band_LOW_PAIRED__Z2
        - aux_cap_helper__b6,r11,band_LOW_PAIRED__condX2 <= 0
 c3473: aux_cap_b2,r1,band_UNPAIRED - aux_c_b2,r1
        + aux_cap_b2,r1,band_LOW_PAIRED + aux_cap_b2,r1,band_HIGH_PAIRED  = 0
 c3474: aux_cap_helper__b9,r6,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b9,r6,band_HIGH_PAIRED__Z2 <= 0
 c3475: aux_cap_helper__b7,r12,band_LOW_PAIRED__condX1
        - aux_cap_helper__b7,r12,band_LOW_PAIRED__Z1 <= 0
 c3476: - sv_function_helpervar_b3,r11__condX1 <= 0
 c3477: - sv_function_helpervar_b5,r4__condX2
        + 0.159946187436471 sv_function_helpervar_b5,r4__Z2 <= 0
 c3478: - 15.9946187432626 sv_function_helpervar_b7,r1__Z3
        + sv_function_helpervar_b7,r1__condX3 <= 0
 c3479: - aux_c_b7,r13 + aux_cap_b7,r13,band_LOW_PAIRED
        + aux_cap_b7,r13,band_HIGH_PAIRED + aux_cap_b7,r13,band_UNPAIRED  = 0
 c3480: aux_cap_helper__b7,r6,band_LOW_PAIRED__Z2
        - aux_cap_helper__b7,r6,band_LOW_PAIRED__condX2 <= 0
 c3481: 523.704262570383 sv_function_helpervar_b7,r0__Z3
        + 285.291690584857 sv_function_helpervar_b7,r0__Z2 - aux_sv_b7,r0
        + 1850.3233249757 sv_function_helpervar_b7,r0__condX1
        + 66.6503596619 sv_function_helpervar_b7,r0__condX2
        + 35.7879648975 sv_function_helpervar_b7,r0__condX3  = 0
 c3482: sv_function_helpervar_b1,r0__Z2 + sv_function_helpervar_b1,r0__Z3
        + sv_function_helpervar_b1,r0__Z1  = 1
 c3483: - sv_function_helpervar_b9,r8__condX1 <= 0
 c3484: aux_cap_b8,r6,band_LOW_PAIRED + aux_cap_b8,r6,band_UNPAIRED
        - aux_c_b8,r6 + aux_cap_b8,r6,band_HIGH_PAIRED  = 0
 c3485: - aux_cap_helper__b1,r4,band_LOW_PAIRED__condX2
        + aux_cap_helper__b1,r4,band_LOW_PAIRED__Z2 <= 0
 c3486: 0.01314 aux_Omega_b3,r11 + 0.00552 aux_Omega_b3,r12
        + 0.01314 aux_Omega_b3,r13 + 0.03125 aux_Omega_b3,r10 - v__3
        + 0.17678 aux_Omega_b3,r8 + 0.07433 aux_Omega_b3,r9
        + 0.42045 aux_Omega_b3,r6 + 0.07433 aux_Omega_b3,r7
        + 0.17678 aux_Omega_b3,r4 + 0.07433 aux_Omega_b3,r5
        + 0.17678 aux_Omega_b3,r2 + 0.42045 aux_Omega_b3,r3 + aux_Omega_b3,r0
        + 0.42045 aux_Omega_b3,r1  = 0
 c3487: - 15.9946187432626 sv_function_helpervar_b6,r4__Z3
        + sv_function_helpervar_b6,r4__condX3 <= 0
 c3488: sv_function_helpervar_b6,r3__condX2
        - 6.06587429996656 sv_function_helpervar_b6,r3__Z2 <= 0
 c3489: 1132.13170491831 aux_sv_b9,r10 - aux_Omega_b9,r10  = 0
 c3490: 2 aux_cap_helper__b9,r0,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b9,r0,band_HIGH_PAIRED__condX3 <= 0
 c3491: X__b9,r0,band_HIGH_PAIRED + X__b9,r0,band_LOW_PAIRED
        - GlobalBidder_W_ir__b9,r0 + X__b9,r0,band_UNPAIRED >= 0
 c3492: - aux_cap_helper__b1,r11,band_UNPAIRED__condX2
        + aux_cap_helper__b1,r11,band_UNPAIRED__Z2 <= 0
 c3493: - aux_cap_b4,r3,band_LOW_PAIRED
        + 3.735962531 aux_cap_helper__b4,r3,band_LOW_PAIRED__condX1
        + 5.2303475434 aux_cap_helper__b4,r3,band_LOW_PAIRED__condX2
        - 1.4943850124 aux_cap_helper__b4,r3,band_LOW_PAIRED__Z2  = 0
 c3494: - aux_cap_helper__b3,r7,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b3,r7,band_HIGH_PAIRED__Z3 <= 0
 c3495: - 0.159946187443648 sv_function_helpervar_b7,r1__Z1
        + sv_function_helpervar_b7,r1__condX1 <= 0
 c3496: 5.45409675243343 sv_function_helpervar_b4,r12__Z3
        - sv_function_helpervar_b4,r12__condX3 <= 0
 c3497: aux_cap_helper__b9,r10,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b9,r10,band_HIGH_PAIRED__condX2 <= 0
 c3498: sv_function_helpervar_b1,r9__condX1
        - 0.159946187442558 sv_function_helpervar_b1,r9__Z1 <= 0
 c3499: - aux_Omega_b5,r0 + 125.476155200051 aux_sv_b5,r0  = 0
 c3500: X__b0,r9,band_UNPAIRED + X__b9,r9,band_UNPAIRED
        + X__b1,r9,band_UNPAIRED + X__b6,r9,band_UNPAIRED
        + X__b5,r9,band_UNPAIRED + X__b7,r9,band_UNPAIRED
        + X__b4,r9,band_UNPAIRED + X__b2,r9,band_UNPAIRED
        + X__b8,r9,band_UNPAIRED + X__b3,r9,band_UNPAIRED <= 2
 c3501: - 2 aux_cap_helper__b2,r9,band_UNPAIRED__Z2
        + aux_cap_helper__b2,r9,band_UNPAIRED__condX2 <= 0
 c3502: aux_cap_helper__b4,r0,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b4,r0,band_UNPAIRED__Z2 <= 0
 c3503: aux_cap_helper__b2,r4,band_UNPAIRED__Z2
        + aux_cap_helper__b2,r4,band_UNPAIRED__Z1  = 1
 c3504: aux_cap_helper__b8,r11,band_LOW_PAIRED__condX1
        - aux_cap_helper__b8,r11,band_LOW_PAIRED__Z1 <= 0
 c3505: 0.8035888618 aux_cap_helper__b8,r1,band_UNPAIRED__condX1
        + 1.1250244065 aux_cap_helper__b8,r1,band_UNPAIRED__condX2
        - aux_cap_b8,r1,band_UNPAIRED
        - 0.3214355447 aux_cap_helper__b8,r1,band_UNPAIRED__Z2  = 0
 c3506: sv_function_helpervar_b3,r10__condX3
        - 15.9946187432626 sv_function_helpervar_b3,r10__Z3 <= 0
 c3507: - X__b2,r1,band_UNPAIRED + aux_cap_helper__b2,r1,band_UNPAIRED__condX1
        + aux_cap_helper__b2,r1,band_UNPAIRED__condX2  = 0
 c3508: 2 aux_cap_helper__b2,r11,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b2,r11,band_HIGH_PAIRED__condX3 <= 0
 c3509: 0.8035888618 aux_cap_helper__b5,r7,band_UNPAIRED__condX1
        + 1.1250244065 aux_cap_helper__b5,r7,band_UNPAIRED__condX2
        - aux_cap_b5,r7,band_UNPAIRED
        - 0.3214355447 aux_cap_helper__b5,r7,band_UNPAIRED__Z2  = 0
 c3510: - aux_cap_helper__b5,r10,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b5,r10,band_HIGH_PAIRED__condX1 <= 0
 c3511: - 2 aux_cap_helper__b2,r8,band_UNPAIRED__Z2
        + aux_cap_helper__b2,r8,band_UNPAIRED__condX2 <= 0
 c3512: sv_function_helpervar_b7,r13__condX2
        - 6.8709716173417 sv_function_helpervar_b7,r13__Z2 <= 0
 c3513: - aux_cap_helper__b2,r11,band_LOW_PAIRED__Z1
        + aux_cap_helper__b2,r11,band_LOW_PAIRED__condX1 <= 0
 c3514: aux_cap_helper__b9,r4,band_UNPAIRED__Z2
        - aux_cap_helper__b9,r4,band_UNPAIRED__condX2 <= 0
 c3515: aux_cap_helper__b5,r13,band_LOW_PAIRED__Z1
        + aux_cap_helper__b5,r13,band_LOW_PAIRED__Z2  = 1
 c3516: aux_cap_helper__b0,r0,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b0,r0,band_HIGH_PAIRED__condX2 <= 0
 c3517: - 2 aux_cap_helper__b1,r12,band_LOW_PAIRED__Z2
        + aux_cap_helper__b1,r12,band_LOW_PAIRED__condX2 <= 0
 c3518: aux_cap_helper__b3,r13,band_UNPAIRED__condX1
        - aux_cap_helper__b3,r13,band_UNPAIRED__Z1 <= 0
 c3519: - aux_cap_helper__b8,r4,band_HIGH_PAIRED__condX1 <= 0
 c3520: - 1.4943850124 aux_cap_helper__b3,r11,band_LOW_PAIRED__Z2
        - aux_cap_b3,r11,band_LOW_PAIRED
        + 5.2303475434 aux_cap_helper__b3,r11,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b3,r11,band_LOW_PAIRED__condX1  = 0
 c3521: - sv_function_helpervar_b1,r5__condX1 <= 0
 c3522: - 0.159946187475059 sv_function_helpervar_b5,r13__Z1
        + sv_function_helpervar_b5,r13__condX1 <= 0
 c3523: aux_cap_helper__b6,r12,band_UNPAIRED__Z2
        - aux_cap_helper__b6,r12,band_UNPAIRED__condX2 <= 0
 c3524: - aux_cap_helper__b2,r4,band_HIGH_PAIRED__condX1 <= 0
 c3525: - aux_cap_helper__b7,r10,band_LOW_PAIRED__condX2
        + aux_cap_helper__b7,r10,band_LOW_PAIRED__Z2 <= 0
 c3526: aux_cap_helper__b9,r0,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b9,r0,band_HIGH_PAIRED__condX2 <= 0
 c3527: - X__b8,r13,band_UNPAIRED
        + aux_cap_helper__b8,r13,band_UNPAIRED__condX1
        + aux_cap_helper__b8,r13,band_UNPAIRED__condX2  = 0
 c3528: 2 aux_cap_helper__b0,r0,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b0,r0,band_HIGH_PAIRED__condX3 <= 0
 c3529: aux_cap_helper__b4,r1,band_UNPAIRED__condX1
        - aux_cap_helper__b4,r1,band_UNPAIRED__Z1 <= 0
 c3530: - aux_cap_helper__b8,r8,band_UNPAIRED__condX2
        + aux_cap_helper__b8,r8,band_UNPAIRED__Z2 <= 0
 c3531: aux_cap_helper__b3,r13,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b3,r13,band_UNPAIRED__Z2 <= 0
 c3532: - aux_cap_helper__b7,r13,band_HIGH_PAIRED__condX1 <= 0
 c3533: aux_cap_helper__b1,r10,band_UNPAIRED__Z1
        + aux_cap_helper__b1,r10,band_UNPAIRED__Z2  = 1
 c3534: - sv_function_helpervar_b2,r2__condX1 <= 0
 c3535: - aux_cap_helper__b2,r6,band_LOW_PAIRED__condX2
        + aux_cap_helper__b2,r6,band_LOW_PAIRED__Z2 <= 0
 c3536: - aux_cap_helper__b2,r9,band_UNPAIRED__Z1
        + aux_cap_helper__b2,r9,band_UNPAIRED__condX1 <= 0
 c3537: aux_cap_helper__b7,r12,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b7,r12,band_LOW_PAIRED__Z2 <= 0
 c3538: 16.7990266453618 sv_function_helpervar_b0,r13__Z2
        + 35.9609256364819 sv_function_helpervar_b0,r13__Z3
        + 109.8706625908 sv_function_helpervar_b0,r13__condX1
        + 4.8414217453 sv_function_helpervar_b0,r13__condX2
        + 1.820969776 sv_function_helpervar_b0,r13__condX3 - aux_sv_b0,r13  = 0
 c3539: sv_function_helpervar_b2,r1__Z2 + sv_function_helpervar_b2,r1__Z3
        + sv_function_helpervar_b2,r1__Z1  = 1
 c3540: sv_function_helpervar_b2,r1__condX3
        - 15.9946187432626 sv_function_helpervar_b2,r1__Z3 <= 0
 c3541: - sv_function_helpervar_b3,r3__condX2
        + 0.159946187517238 sv_function_helpervar_b3,r3__Z2 <= 0
 c3542: 25.683814777 sv_function_helpervar_b8,r10__condX3
        + 48.9850633646 sv_function_helpervar_b8,r10__condX2
        + 1343.0085755194 sv_function_helpervar_b8,r10__condX1 - aux_sv_b8,r10
        + 206.974127225905 sv_function_helpervar_b8,r10__Z2
        + 384.786439038973 sv_function_helpervar_b8,r10__Z3  = 0
 c3543: aux_cap_helper__b2,r11,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b2,r11,band_HIGH_PAIRED__condX2 <= 0
 c3544: 272.89077033813 aux_sv_b8,r8 - aux_Omega_b8,r8  = 0
 c3545: aux_cap_helper__b4,r9,band_LOW_PAIRED__condX1
        + aux_cap_helper__b4,r9,band_LOW_PAIRED__condX2
        - X__b4,r9,band_LOW_PAIRED  = 0
 c3546: - sv_function_helpervar_b5,r9__condX1 <= 0
 c3547: aux_cap_b2,r12,band_LOW_PAIRED + aux_cap_b2,r12,band_UNPAIRED
        + aux_cap_b2,r12,band_HIGH_PAIRED - aux_c_b2,r12  = 0
 c3548: aux_cap_helper__b8,r10,band_UNPAIRED__Z2
        - aux_cap_helper__b8,r10,band_UNPAIRED__condX2 <= 0
 c3549: X__b9,r3,band_LOW_PAIRED + X__b9,r3,band_UNPAIRED
        - GlobalBidder_W_ir__b9,r3 + X__b9,r3,band_HIGH_PAIRED >= 0
 c3550: - aux_sv_b0,r0 + 35.4584296569887 sv_function_helpervar_b0,r0__Z3
        + 16.8189932903473 sv_function_helpervar_b0,r0__Z2
        + 1.852386341 sv_function_helpervar_b0,r0__condX3
        + 4.7165882293 sv_function_helpervar_b0,r0__condX2
        + 109.8706625966 sv_function_helpervar_b0,r0__condX1  = 0
 c3551: aux_cap_helper__b5,r7,band_LOW_PAIRED__Z2
        + aux_cap_helper__b5,r7,band_LOW_PAIRED__Z1  = 1
 c3552: sv_function_helpervar_b7,r11__Z1 + sv_function_helpervar_b7,r11__Z3
        + sv_function_helpervar_b7,r11__Z2  = 1
 c3553: - 2 aux_cap_helper__b2,r12,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b2,r12,band_HIGH_PAIRED__condX2 <= 0
 c3554: X__b7,r7,band_UNPAIRED + X__b7,r7,band_LOW_PAIRED
        - GlobalBidder_W_ir__b7,r7 + X__b7,r7,band_HIGH_PAIRED >= 0
 c3555: aux_cap_helper__b5,r0,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b5,r0,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b5,r0,band_HIGH_PAIRED__Z3  = 1
 c3556: - 3 aux_cap_helper__b6,r9,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b6,r9,band_HIGH_PAIRED__condX3 <= 0
 c3557: 2 aux_cap_helper__b3,r0,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b3,r0,band_HIGH_PAIRED__condX3 <= 0
 c3558: 1.5453622427 aux_cap_helper__b3,r13,band_HIGH_PAIRED__condX1
        + 2.1635071398 aux_cap_helper__b3,r13,band_HIGH_PAIRED__condX2
        + 1.3908260184 aux_cap_helper__b3,r13,band_HIGH_PAIRED__condX3
        - 0.6181448971 aux_cap_helper__b3,r13,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b3,r13,band_HIGH_PAIRED__Z3
        - aux_cap_b3,r13,band_HIGH_PAIRED  = 0
 c3559: - aux_cap_helper__b1,r3,band_HIGH_PAIRED__condX1 <= 0
 c3560: - 2 aux_cap_helper__b5,r10,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b5,r10,band_HIGH_PAIRED__condX2 <= 0
 c3561: sv_function_helpervar_b1,r6__Z1 + sv_function_helpervar_b1,r6__Z3
        + sv_function_helpervar_b1,r6__Z2  = 1
 c3562: - 15.9946187432626 sv_function_helpervar_b9,r12__Z3
        + sv_function_helpervar_b9,r12__condX3 <= 0
 c3563: aux_cap_b5,r10,band_HIGH_PAIRED + aux_cap_b5,r10,band_UNPAIRED
        - aux_c_b5,r10 + aux_cap_b5,r10,band_LOW_PAIRED  = 0
 c3564: - aux_cap_helper__b4,r2,band_LOW_PAIRED__Z1
        + aux_cap_helper__b4,r2,band_LOW_PAIRED__condX1 <= 0
 c3565: aux_cap_helper__b6,r6,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b6,r6,band_HIGH_PAIRED__Z2 <= 0
 c3566: - aux_cap_helper__b1,r2,band_UNPAIRED__condX1 <= 0
 c3567: sv_function_helpervar_b3,r10__Z3 + sv_function_helpervar_b3,r10__Z1
        + sv_function_helpervar_b3,r10__Z2  = 1
 c3568: - aux_cap_helper__b0,r7,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b0,r7,band_HIGH_PAIRED__Z3 <= 0
 c3569: - sv_function_helpervar_b6,r5__condX1 <= 0
 c3570: GlobalBidder_W_ir__b7,r3 + GlobalBidder_W_ir__b7,r2
        + GlobalBidder_W_ir__b7,r1 + GlobalBidder_W_ir__b7,r0
        + GlobalBidder_W_ir__b7,r7 + GlobalBidder_W_ir__b7,r6
        + GlobalBidder_W_ir__b7,r5 + GlobalBidder_W_ir__b7,r4
        + GlobalBidder_W_ir__b7,r9 + GlobalBidder_W_ir__b7,r8
        - GlobalBidder_W_i_7 + GlobalBidder_W_ir__b7,r13
        + GlobalBidder_W_ir__b7,r11 + GlobalBidder_W_ir__b7,r12
        + GlobalBidder_W_ir__b7,r10  = 0
 c3571: aux_cap_helper__b8,r2,band_UNPAIRED__Z1
        + aux_cap_helper__b8,r2,band_UNPAIRED__Z2  = 1
 c3572: 411.987911574017 sv_function_helpervar_b8,r1__Z3
        + 206.298541451196 sv_function_helpervar_b8,r1__Z2
        + 53.2088950391 sv_function_helpervar_b8,r1__condX2
        + 23.9831507617 sv_function_helpervar_b8,r1__condX3
        + 1343.0085752047 sv_function_helpervar_b8,r1__condX1 - aux_sv_b8,r1
         = 0
 c3573: - 2 aux_cap_helper__b3,r6,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b3,r6,band_HIGH_PAIRED__condX2 <= 0
 c3574: aux_cap_helper__b5,r6,band_LOW_PAIRED__Z2
        + aux_cap_helper__b5,r6,band_LOW_PAIRED__Z1  = 1
 c3575: X__b8,r5,band_HIGH_PAIRED + X__b8,r5,band_LOW_PAIRED
        + X__b8,r5,band_UNPAIRED - GlobalBidder_W_ir__b8,r5 >= 0
 c3576: aux_cap_helper__b0,r7,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b0,r7,band_LOW_PAIRED__Z2 <= 0
 c3577: aux_cap_helper__b2,r5,band_UNPAIRED__Z2
        - aux_cap_helper__b2,r5,band_UNPAIRED__condX2 <= 0
 c3578: 3695.48313578368 aux_sv_b8,r4 - aux_Omega_b8,r4  = 0
 c3579: - aux_cap_helper__b2,r12,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b2,r12,band_HIGH_PAIRED__condX1 <= 0
 c3580: sv_function_helpervar_b1,r7__condX1
        - 0.159946187439929 sv_function_helpervar_b1,r7__Z1 <= 0
 c3581: - aux_cap_helper__b7,r11,band_UNPAIRED__condX1 <= 0
 c3582: aux_cap_helper__b9,r9,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b9,r9,band_HIGH_PAIRED__Z3 <= 0
 c3583: - 0.3214355447 aux_cap_helper__b9,r9,band_UNPAIRED__Z2
        + 0.8035888618 aux_cap_helper__b9,r9,band_UNPAIRED__condX1
        + 1.1250244065 aux_cap_helper__b9,r9,band_UNPAIRED__condX2
        - aux_cap_b9,r9,band_UNPAIRED  = 0
 c3584: - aux_cap_helper__b8,r5,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b8,r5,band_HIGH_PAIRED__condX1 <= 0
 c3585: 6.35577727875249 sv_function_helpervar_b3,r6__Z3
        - sv_function_helpervar_b3,r6__condX3 <= 0
 c3586: - 2 aux_cap_helper__b1,r6,band_LOW_PAIRED__Z2
        + aux_cap_helper__b1,r6,band_LOW_PAIRED__condX2 <= 0
 c3587: - X__b1,r10,band_UNPAIRED
        + aux_cap_helper__b1,r10,band_UNPAIRED__condX2
        + aux_cap_helper__b1,r10,band_UNPAIRED__condX1  = 0
 c3588: sv_function_helpervar_b3,r0__condX1
        - 0.159946187458717 sv_function_helpervar_b3,r0__Z1 <= 0
 c3589: sv_function_helpervar_b9,r7__condX3
        - 15.9946187432626 sv_function_helpervar_b9,r7__Z3 <= 0
 c3590: aux_cap_helper__b0,r4,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b0,r4,band_HIGH_PAIRED__Z3 <= 0
 c3591: sv_function_helpervar_b7,r13__Z1 + sv_function_helpervar_b7,r13__Z2
        + sv_function_helpervar_b7,r13__Z3  = 1
 c3592: - X__b1,r3,band_HIGH_PAIRED
        + aux_cap_helper__b1,r3,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b1,r3,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b1,r3,band_HIGH_PAIRED__condX1  = 0
 c3593: - aux_cap_helper__b0,r6,band_UNPAIRED__condX2
        + aux_cap_helper__b0,r6,band_UNPAIRED__Z2 <= 0
 c3594: - aux_cap_helper__b4,r3,band_HIGH_PAIRED__condX1 <= 0
 c3595: - 3 aux_cap_helper__b4,r12,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b4,r12,band_HIGH_PAIRED__condX3 <= 0
 c3596: - 2 aux_cap_helper__b8,r5,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b8,r5,band_HIGH_PAIRED__condX2 <= 0
 c3597: - aux_cap_b9,r12,band_UNPAIRED
        - 0.3214355447 aux_cap_helper__b9,r12,band_UNPAIRED__Z2
        + 0.8035888618 aux_cap_helper__b9,r12,band_UNPAIRED__condX1
        + 1.1250244065 aux_cap_helper__b9,r12,band_UNPAIRED__condX2  = 0
 c3598: sv_function_helpervar_b8,r13__condX1
        + sv_function_helpervar_b8,r13__condX2
        + sv_function_helpervar_b8,r13__condX3 - aux_c_b8,r13  = 0
 c3599: - aux_c_b5,r11 + sv_function_helpervar_b5,r11__condX1
        + sv_function_helpervar_b5,r11__condX3
        + sv_function_helpervar_b5,r11__condX2  = 0
 c3600: aux_cap_helper__b4,r12,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b4,r12,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b4,r12,band_HIGH_PAIRED__Z1  = 1
 c3601: 1.1250244065 aux_cap_helper__b3,r12,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b3,r12,band_UNPAIRED__condX1
        - 0.3214355447 aux_cap_helper__b3,r12,band_UNPAIRED__Z2
        - aux_cap_b3,r12,band_UNPAIRED  = 0
 c3602: - aux_cap_helper__b6,r3,band_UNPAIRED__condX1 <= 0
 c3603: - aux_cap_helper__b0,r7,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b0,r7,band_HIGH_PAIRED__Z2 <= 0
 c3604: - 0.3214355447 aux_cap_helper__b8,r12,band_UNPAIRED__Z2
        + 1.1250244065 aux_cap_helper__b8,r12,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b8,r12,band_UNPAIRED__condX1
        - aux_cap_b8,r12,band_UNPAIRED  = 0
 c3605: X__b8,r13,band_UNPAIRED + X__b8,r13,band_LOW_PAIRED
        - GlobalBidder_W_ir__b8,r13 + X__b8,r13,band_HIGH_PAIRED >= 0
 c3606: sv_function_helpervar_b8,r6__condX2
        - 6.94024412210506 sv_function_helpervar_b8,r6__Z2 <= 0
 c3607: 0.159946187445724 sv_function_helpervar_b9,r8__Z2
        - sv_function_helpervar_b9,r8__condX2 <= 0
 c3608: - aux_cap_helper__b3,r7,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b3,r7,band_HIGH_PAIRED__Z2 <= 0
 c3609: - aux_cap_helper__b5,r7,band_UNPAIRED__condX2
        + aux_cap_helper__b5,r7,band_UNPAIRED__Z2 <= 0
 c3610: - 3 aux_cap_helper__b3,r9,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b3,r9,band_HIGH_PAIRED__condX3 <= 0
 c3611: aux_cap_helper__b0,r9,band_LOW_PAIRED__Z1
        + aux_cap_helper__b0,r9,band_LOW_PAIRED__Z2  = 1
 c3612: - 2 aux_cap_helper__b0,r13,band_LOW_PAIRED__Z2
        + aux_cap_helper__b0,r13,band_LOW_PAIRED__condX2 <= 0
 c3613: - aux_c_b7,r8 + aux_cap_b7,r8,band_UNPAIRED
        + aux_cap_b7,r8,band_LOW_PAIRED + aux_cap_b7,r8,band_HIGH_PAIRED  = 0
 c3614: 0.8035888618 aux_cap_helper__b7,r12,band_UNPAIRED__condX1
        + 1.1250244065 aux_cap_helper__b7,r12,band_UNPAIRED__condX2
        - aux_cap_b7,r12,band_UNPAIRED
        - 0.3214355447 aux_cap_helper__b7,r12,band_UNPAIRED__Z2  = 0
 c3615: aux_cap_helper__b8,r5,band_LOW_PAIRED__Z2
        + aux_cap_helper__b8,r5,band_LOW_PAIRED__Z1  = 1
 c3616: - sv_function_helpervar_b8,r13__condX1 <= 0
 c3617: - aux_cap_helper__b1,r6,band_LOW_PAIRED__Z1
        + aux_cap_helper__b1,r6,band_LOW_PAIRED__condX1 <= 0
 c3618: - aux_cap_b3,r8,band_HIGH_PAIRED
        + 0.927217345653617 aux_cap_helper__b3,r8,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b3,r8,band_HIGH_PAIRED__Z2
        + 1.3908260184 aux_cap_helper__b3,r8,band_HIGH_PAIRED__condX3
        + 1.5453622427 aux_cap_helper__b3,r8,band_HIGH_PAIRED__condX1
        + 2.1635071398 aux_cap_helper__b3,r8,band_HIGH_PAIRED__condX2  = 0
 c3619: - aux_Omega_b2,r9 + 249.920598793581 aux_sv_b2,r9  = 0
 c3620: aux_cap_helper__b9,r8,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b9,r8,band_UNPAIRED__Z2 <= 0
 c3621: aux_cap_helper__b6,r10,band_UNPAIRED__Z1
        + aux_cap_helper__b6,r10,band_UNPAIRED__Z2  = 1
 c3622: - aux_cap_helper__b5,r0,band_UNPAIRED__condX1 <= 0
 c3623: 2 aux_cap_helper__b1,r10,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b1,r10,band_HIGH_PAIRED__condX3 <= 0
 c3624: - 0.159946187454416 sv_function_helpervar_b2,r9__Z1
        + sv_function_helpervar_b2,r9__condX1 <= 0
 c3625: aux_cap_helper__b6,r7,band_LOW_PAIRED__condX2
        + aux_cap_helper__b6,r7,band_LOW_PAIRED__condX1
        - X__b6,r7,band_LOW_PAIRED  = 0
 c3626: - aux_cap_helper__b6,r1,band_LOW_PAIRED__condX1 <= 0
 c3627: - aux_cap_helper__b7,r3,band_HIGH_PAIRED__condX1 <= 0
 c3628: aux_cap_helper__b9,r3,band_UNPAIRED__Z1
        + aux_cap_helper__b9,r3,band_UNPAIRED__Z2  = 1
 c3629: - 0.159946187432792 sv_function_helpervar_b1,r3__Z1
        + sv_function_helpervar_b1,r3__condX1 <= 0
 c3630: - aux_cap_helper__b9,r7,band_LOW_PAIRED__condX1 <= 0
 c3631: 5.2303475434 aux_cap_helper__b1,r11,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b1,r11,band_LOW_PAIRED__condX1
        - 1.4943850124 aux_cap_helper__b1,r11,band_LOW_PAIRED__Z2
        - aux_cap_b1,r11,band_LOW_PAIRED  = 0
 c3632: 5.2303475434 aux_cap_helper__b4,r5,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b4,r5,band_LOW_PAIRED__condX1
        - aux_cap_b4,r5,band_LOW_PAIRED
        - 1.4943850124 aux_cap_helper__b4,r5,band_LOW_PAIRED__Z2  = 0
 c3633: - aux_sv_b5,r10 + 1169.5974419688 sv_function_helpervar_b5,r10__condX1
        + 55.8657142637 sv_function_helpervar_b5,r10__condX2
        + 18.4678692393 sv_function_helpervar_b5,r10__condX3
        + 178.137143740681 sv_function_helpervar_b5,r10__Z2
        + 397.475145661653 sv_function_helpervar_b5,r10__Z3  = 0
 c3634: - aux_sv_b0,r1 + 35.8774723118111 sv_function_helpervar_b0,r1__Z3
        + 16.8024639280141 sv_function_helpervar_b0,r1__Z2
        + 109.8706625477 sv_function_helpervar_b0,r1__condX1
        + 1.8261873636 sv_function_helpervar_b0,r1__condX3
        + 4.819931499 sv_function_helpervar_b0,r1__condX2  = 0
 c3635: - v__7 + 0.94875 GlobalBidder_PSI__i7,k2
        + 0.98719 GlobalBidder_PSI__i7,k1 + GlobalBidder_PSI__i7,k0  = 0
 c3636: aux_cap_helper__b5,r12,band_UNPAIRED__Z2
        + aux_cap_helper__b5,r12,band_UNPAIRED__Z1  = 1
 c3637: - aux_cap_helper__b3,r5,band_LOW_PAIRED__condX1 <= 0
 c3638: - sv_function_helpervar_b4,r8__condX1 <= 0
 c3639: aux_cap_helper__b7,r9,band_UNPAIRED__condX1
        - aux_cap_helper__b7,r9,band_UNPAIRED__Z1 <= 0
 c3640: - 2 aux_cap_helper__b5,r10,band_UNPAIRED__Z2
        + aux_cap_helper__b5,r10,band_UNPAIRED__condX2 <= 0
 c3641: - aux_cap_helper__b5,r5,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b5,r5,band_HIGH_PAIRED__condX1 <= 0
 c3642: aux_cap_helper__b8,r1,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b8,r1,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b8,r1,band_HIGH_PAIRED__condX3
        - X__b8,r1,band_HIGH_PAIRED  = 0
 c3643: aux_cap_helper__b3,r10,band_LOW_PAIRED__Z2
        + aux_cap_helper__b3,r10,band_LOW_PAIRED__Z1  = 1
 c3644: 1505.33616201499 aux_sv_b4,r11 - aux_Omega_b4,r11  = 0
 c3645: sv_function_helpervar_b4,r2__condX1
        - 0.159946187453323 sv_function_helpervar_b4,r2__Z1 <= 0
 c3646: - aux_cap_helper__b6,r12,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b6,r12,band_HIGH_PAIRED__Z2 <= 0
 c3647: aux_cap_helper__b7,r3,band_UNPAIRED__condX1
        + aux_cap_helper__b7,r3,band_UNPAIRED__condX2 - X__b7,r3,band_UNPAIRED
         = 0
 c3648: sv_function_helpervar_b9,r0__Z1 + sv_function_helpervar_b9,r0__Z2
        + sv_function_helpervar_b9,r0__Z3  = 1
 c3649: X__b4,r0,band_UNPAIRED + X__b3,r0,band_UNPAIRED
        + X__b5,r0,band_UNPAIRED + X__b1,r0,band_UNPAIRED
        + X__b7,r0,band_UNPAIRED + X__b2,r0,band_UNPAIRED
        + X__b6,r0,band_UNPAIRED + X__b0,r0,band_UNPAIRED
        + X__b8,r0,band_UNPAIRED + X__b9,r0,band_UNPAIRED <= 2
 c3650: - 15.9946187432626 sv_function_helpervar_b4,r7__Z3
        + sv_function_helpervar_b4,r7__condX3 <= 0
 c3651: aux_cap_helper__b7,r11,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b7,r11,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b7,r11,band_HIGH_PAIRED__condX2
        - X__b7,r11,band_HIGH_PAIRED  = 0
 c3652: aux_cap_helper__b8,r0,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b8,r0,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b8,r0,band_HIGH_PAIRED__Z1  = 1
 c3653: - aux_Omega_b9,r8 + 499.566374455272 aux_sv_b9,r8  = 0
 c3654: aux_cap_helper__b9,r1,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b9,r1,band_HIGH_PAIRED__Z1 <= 0
 c3655: - aux_cap_helper__b0,r13,band_LOW_PAIRED__Z1
        + aux_cap_helper__b0,r13,band_LOW_PAIRED__condX1 <= 0
 c3656: aux_cap_helper__b9,r8,band_UNPAIRED__condX1
        - aux_cap_helper__b9,r8,band_UNPAIRED__Z1 <= 0
 c3657: - X__b4,r1,band_HIGH_PAIRED
        + aux_cap_helper__b4,r1,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b4,r1,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b4,r1,band_HIGH_PAIRED__condX3  = 0
 c3658: aux_cap_helper__b1,r0,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b1,r0,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b1,r0,band_HIGH_PAIRED__Z1  = 1
 c3659: aux_cap_helper__b1,r9,band_UNPAIRED__Z2
        - aux_cap_helper__b1,r9,band_UNPAIRED__condX2 <= 0
 c3660: aux_cap_helper__b1,r10,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b1,r10,band_HIGH_PAIRED__condX2 <= 0
 c3661: - aux_cap_helper__b3,r1,band_UNPAIRED__condX1 <= 0
 c3662: aux_cap_helper__b8,r5,band_LOW_PAIRED__Z2
        - aux_cap_helper__b8,r5,band_LOW_PAIRED__condX2 <= 0
 c3663: - 0.159946187471245 sv_function_helpervar_b6,r8__Z1
        + sv_function_helpervar_b6,r8__condX1 <= 0
 c3664: aux_cap_helper__b1,r13,band_UNPAIRED__condX2
        + aux_cap_helper__b1,r13,band_UNPAIRED__condX1
        - X__b1,r13,band_UNPAIRED  = 0
 c3665: - aux_cap_helper__b3,r8,band_UNPAIRED__condX2
        + aux_cap_helper__b3,r8,band_UNPAIRED__Z2 <= 0
 c3666: - sv_function_helpervar_b5,r0__condX2
        + 0.159946187449403 sv_function_helpervar_b5,r0__Z2 <= 0
 c3667: aux_cap_helper__b9,r11,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b9,r11,band_HIGH_PAIRED__Z2 <= 0
 c3668: X__b5,r8,band_LOW_PAIRED + X__b1,r8,band_LOW_PAIRED
        + X__b3,r8,band_LOW_PAIRED + X__b0,r8,band_LOW_PAIRED
        + X__b2,r8,band_LOW_PAIRED + X__b4,r8,band_LOW_PAIRED
        + X__b8,r8,band_LOW_PAIRED + X__b9,r8,band_LOW_PAIRED
        + X__b6,r8,band_LOW_PAIRED + X__b7,r8,band_LOW_PAIRED <= 2
 c3669: - aux_cap_b2,r5,band_UNPAIRED
        - 0.3214355447 aux_cap_helper__b2,r5,band_UNPAIRED__Z2
        + 1.1250244065 aux_cap_helper__b2,r5,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b2,r5,band_UNPAIRED__condX1  = 0
 c3670: X__b5,r7,band_HIGH_PAIRED + X__b2,r7,band_HIGH_PAIRED
        + X__b1,r7,band_HIGH_PAIRED + X__b0,r7,band_HIGH_PAIRED
        + X__b3,r7,band_HIGH_PAIRED + X__b4,r7,band_HIGH_PAIRED
        + X__b6,r7,band_HIGH_PAIRED + X__b9,r7,band_HIGH_PAIRED
        + X__b7,r7,band_HIGH_PAIRED + X__b8,r7,band_HIGH_PAIRED <= 3
 c3671: 0.159946187547474 sv_function_helpervar_b6,r10__Z2
        - sv_function_helpervar_b6,r10__condX2 <= 0
 c3672: - 1.4943850124 aux_cap_helper__b9,r0,band_LOW_PAIRED__Z2
        - aux_cap_b9,r0,band_LOW_PAIRED
        + 3.735962531 aux_cap_helper__b9,r0,band_LOW_PAIRED__condX1
        + 5.2303475434 aux_cap_helper__b9,r0,band_LOW_PAIRED__condX2  = 0
 c3673: sv_function_helpervar_b6,r3__condX1
        - 0.159946187469737 sv_function_helpervar_b6,r3__Z1 <= 0
 c3674: - X__b1,r5,band_UNPAIRED + aux_cap_helper__b1,r5,band_UNPAIRED__condX2
        + aux_cap_helper__b1,r5,band_UNPAIRED__condX1  = 0
 c3675: aux_cap_helper__b1,r1,band_LOW_PAIRED__Z2
        - aux_cap_helper__b1,r1,band_LOW_PAIRED__condX2 <= 0
 c3676: - sv_function_helpervar_b1,r6__condX3
        + 6.60983461492046 sv_function_helpervar_b1,r6__Z3 <= 0
 c3677: aux_cap_helper__b2,r3,band_LOW_PAIRED__Z2
        - aux_cap_helper__b2,r3,band_LOW_PAIRED__condX2 <= 0
 c3678: - 15.9946187432626 sv_function_helpervar_b3,r6__Z3
        + sv_function_helpervar_b3,r6__condX3 <= 0
 c3679: - aux_cap_helper__b4,r4,band_UNPAIRED__condX1 <= 0
 c3680: - X__b5,r8,band_LOW_PAIRED
        + aux_cap_helper__b5,r8,band_LOW_PAIRED__condX1
        + aux_cap_helper__b5,r8,band_LOW_PAIRED__condX2  = 0
 c3681: aux_cap_helper__b9,r3,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b9,r3,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b9,r3,band_HIGH_PAIRED__condX1
        - X__b9,r3,band_HIGH_PAIRED  = 0
 c3682: 1.3908260184 aux_cap_helper__b1,r6,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b1,r6,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b1,r6,band_HIGH_PAIRED__condX1
        + 0.927217345653617 aux_cap_helper__b1,r6,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b1,r6,band_HIGH_PAIRED__Z2
        - aux_cap_b1,r6,band_HIGH_PAIRED  = 0
 c3683: aux_cap_helper__b4,r0,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b4,r0,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b4,r0,band_HIGH_PAIRED__Z1  = 1
 c3684: - sv_function_helpervar_b2,r2__condX3
        + 6.10017409005981 sv_function_helpervar_b2,r2__Z3 <= 0
 c3685: aux_cap_helper__b7,r9,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b7,r9,band_UNPAIRED__Z2 <= 0
 c3686: - X__b0,r2,band_LOW_PAIRED
        + aux_cap_helper__b0,r2,band_LOW_PAIRED__condX2
        + aux_cap_helper__b0,r2,band_LOW_PAIRED__condX1  = 0
 c3687: - 0.159946187518603 sv_function_helpervar_b5,r3__Z1
        + sv_function_helpervar_b5,r3__condX1 <= 0
 c3688: - aux_cap_helper__b6,r12,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b6,r12,band_HIGH_PAIRED__Z3 <= 0
 c3689: - aux_Omega_b7,r12 + 2419.68334363167 aux_sv_b7,r12  = 0
 c3690: - aux_cap_helper__b7,r3,band_LOW_PAIRED__condX2
        + aux_cap_helper__b7,r3,band_LOW_PAIRED__Z2 <= 0
 c3691: aux_cap_helper__b4,r7,band_LOW_PAIRED__Z1
        + aux_cap_helper__b4,r7,band_LOW_PAIRED__Z2  = 1
 c3692: sv_function_helpervar_b5,r5__Z1 + sv_function_helpervar_b5,r5__Z2
        + sv_function_helpervar_b5,r5__Z3  = 1
 c3693: aux_cap_helper__b5,r8,band_UNPAIRED__Z2
        + aux_cap_helper__b5,r8,band_UNPAIRED__Z1  = 1
 c3694: - aux_cap_b8,r2,band_HIGH_PAIRED
        - 0.6181448971 aux_cap_helper__b8,r2,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b8,r2,band_HIGH_PAIRED__Z3
        + 1.3908260184 aux_cap_helper__b8,r2,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b8,r2,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b8,r2,band_HIGH_PAIRED__condX1  = 0
 c3695: - X__b1,r1,band_LOW_PAIRED
        + aux_cap_helper__b1,r1,band_LOW_PAIRED__condX2
        + aux_cap_helper__b1,r1,band_LOW_PAIRED__condX1  = 0
 c3696: 1.5453622427 aux_cap_helper__b0,r10,band_HIGH_PAIRED__condX1
        + 2.1635071398 aux_cap_helper__b0,r10,band_HIGH_PAIRED__condX2
        + 1.3908260184 aux_cap_helper__b0,r10,band_HIGH_PAIRED__condX3
        - aux_cap_b0,r10,band_HIGH_PAIRED
        - 0.6181448971 aux_cap_helper__b0,r10,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b0,r10,band_HIGH_PAIRED__Z3  = 0
 c3697: - aux_c_b1,r3 + sv_function_helpervar_b1,r3__condX1
        + sv_function_helpervar_b1,r3__condX2
        + sv_function_helpervar_b1,r3__condX3  = 0
 c3698: aux_cap_helper__b9,r11,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b9,r11,band_HIGH_PAIRED__Z1 <= 0
 c3699: - 2 aux_cap_helper__b4,r2,band_LOW_PAIRED__Z2
        + aux_cap_helper__b4,r2,band_LOW_PAIRED__condX2 <= 0
 c3700: 1.5453622427 aux_cap_helper__b0,r4,band_HIGH_PAIRED__condX1
        + 1.3908260184 aux_cap_helper__b0,r4,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b0,r4,band_HIGH_PAIRED__condX2
        - aux_cap_b0,r4,band_HIGH_PAIRED
        + 0.927217345653617 aux_cap_helper__b0,r4,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b0,r4,band_HIGH_PAIRED__Z2  = 0
 c3701: 8.21941784309037 sv_function_helpervar_b9,r8__Z3
        - sv_function_helpervar_b9,r8__condX3 <= 0
 c3702: aux_cap_helper__b3,r3,band_UNPAIRED__condX2
        + aux_cap_helper__b3,r3,band_UNPAIRED__condX1 - X__b3,r3,band_UNPAIRED
         = 0
 c3703: 25.1489940078454 aux_sv_b1,r13 - aux_Omega_b1,r13  = 0
 c3704: - sv_function_helpervar_b5,r8__condX2
        + 0.159946187453478 sv_function_helpervar_b5,r8__Z2 <= 0
 c3705: aux_cap_helper__b3,r13,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b3,r13,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b3,r13,band_HIGH_PAIRED__condX3
        - X__b3,r13,band_HIGH_PAIRED  = 0
 c3706: - 15.9946187432626 sv_function_helpervar_b8,r1__Z3
        + sv_function_helpervar_b8,r1__condX3 <= 0
 c3707: - aux_cap_helper__b8,r7,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b8,r7,band_HIGH_PAIRED__condX1 <= 0
 c3708: aux_cap_b3,r2,band_HIGH_PAIRED - aux_c_b3,r2
        + aux_cap_b3,r2,band_UNPAIRED + aux_cap_b3,r2,band_LOW_PAIRED  = 0
 c3709: sv_function_helpervar_b7,r2__condX2
        - 7.30447265021623 sv_function_helpervar_b7,r2__Z2 <= 0
 c3710: - aux_cap_helper__b8,r6,band_LOW_PAIRED__condX2
        + aux_cap_helper__b8,r6,band_LOW_PAIRED__Z2 <= 0
 c3711: aux_cap_helper__b6,r3,band_LOW_PAIRED__Z1
        + aux_cap_helper__b6,r3,band_LOW_PAIRED__Z2  = 1
 c3712: - X__b2,r12,band_HIGH_PAIRED
        + aux_cap_helper__b2,r12,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b2,r12,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b2,r12,band_HIGH_PAIRED__condX3  = 0
 c3713: - sv_function_helpervar_b9,r6__condX3
        + 6.20001223735563 sv_function_helpervar_b9,r6__Z3 <= 0
 c3714: - X__b1,r6,band_HIGH_PAIRED
        + aux_cap_helper__b1,r6,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b1,r6,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b1,r6,band_HIGH_PAIRED__condX1  = 0
 c3715: - sv_function_helpervar_b1,r6__condX1 <= 0
 c3716: - aux_cap_helper__b6,r10,band_UNPAIRED__condX2
        + aux_cap_helper__b6,r10,band_UNPAIRED__Z2 <= 0
 c3717: - aux_cap_helper__b2,r12,band_LOW_PAIRED__condX1 <= 0
 c3718: sv_function_helpervar_b8,r11__condX3
        - 15.9946187432626 sv_function_helpervar_b8,r11__Z3 <= 0
 c3719: aux_cap_helper__b9,r6,band_LOW_PAIRED__Z2
        + aux_cap_helper__b9,r6,band_LOW_PAIRED__Z1  = 1
 c3720: - aux_sv_b0,r9 + 16.7162456903011 sv_function_helpervar_b0,r9__Z2
        + 37.6583754031444 sv_function_helpervar_b0,r9__Z3
        + 109.8706626115 sv_function_helpervar_b0,r9__condX1
        + 1.7148434723 sv_function_helpervar_b0,r9__condX3
        + 5.3589767838 sv_function_helpervar_b0,r9__condX2  = 0
 c3721: 32.9158694432 sv_function_helpervar_b7,r3__condX3
        + 1850.323325944 sv_function_helpervar_b7,r3__condX1
        + 73.6778795608 sv_function_helpervar_b7,r3__condX2
        + 284.167665575468 sv_function_helpervar_b7,r3__Z2
        + 569.642334356063 sv_function_helpervar_b7,r3__Z3 - aux_sv_b7,r3  = 0
 c3722: aux_cap_helper__b0,r2,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b0,r2,band_HIGH_PAIRED__Z3 <= 0
 c3723: aux_cap_helper__b1,r11,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b1,r11,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b1,r11,band_HIGH_PAIRED__condX3
        - X__b1,r11,band_HIGH_PAIRED  = 0
 c3724: aux_cap_helper__b7,r4,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b7,r4,band_HIGH_PAIRED__condX2 <= 0
 c3725: - aux_cap_b6,r13,band_UNPAIRED
        - 0.3214355447 aux_cap_helper__b6,r13,band_UNPAIRED__Z2
        + 0.8035888618 aux_cap_helper__b6,r13,band_UNPAIRED__condX1
        + 1.1250244065 aux_cap_helper__b6,r13,band_UNPAIRED__condX2  = 0
 c3726: aux_cap_helper__b7,r11,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b7,r11,band_HIGH_PAIRED__condX2 <= 0
 c3727: - 7.48967187319432 sv_function_helpervar_b7,r1__Z2
        + sv_function_helpervar_b7,r1__condX2 <= 0
 c3728: - aux_cap_helper__b6,r10,band_UNPAIRED__condX1 <= 0
 c3729: - sv_function_helpervar_b2,r4__condX1 <= 0
 c3730: 0.159946187445632 sv_function_helpervar_b3,r12__Z2
        - sv_function_helpervar_b3,r12__condX2 <= 0
 c3731: - aux_cap_helper__b7,r4,band_HIGH_PAIRED__condX1 <= 0
 c3732: aux_cap_helper__b8,r1,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b8,r1,band_HIGH_PAIRED__Z3 <= 0
 c3733: aux_cap_helper__b0,r8,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b0,r8,band_HIGH_PAIRED__Z1 <= 0
 c3734: - aux_cap_helper__b0,r0,band_UNPAIRED__condX2
        + aux_cap_helper__b0,r0,band_UNPAIRED__Z2 <= 0
 c3735: - aux_cap_helper__b2,r2,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b2,r2,band_HIGH_PAIRED__Z2 <= 0
 c3736: 426.595508263591 sv_function_helpervar_b8,r9__Z3
        + 205.835960021848 sv_function_helpervar_b8,r9__Z2 - aux_sv_b8,r9
        + 1343.0085755386 sv_function_helpervar_b8,r9__condX1
        + 56.1010016837 sv_function_helpervar_b8,r9__condX2
        + 23.0698688058 sv_function_helpervar_b8,r9__condX3  = 0
 c3737: - aux_cap_helper__b7,r11,band_HIGH_PAIRED__condX1 <= 0
 c3738: - aux_cap_helper__b4,r8,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b4,r8,band_HIGH_PAIRED__Z2 <= 0
 c3739: - aux_c_b5,r2 + sv_function_helpervar_b5,r2__condX2
        + sv_function_helpervar_b5,r2__condX3
        + sv_function_helpervar_b5,r2__condX1  = 0
 c3740: X__b4,r7,band_LOW_PAIRED + X__b6,r7,band_LOW_PAIRED
        + X__b2,r7,band_LOW_PAIRED + X__b9,r7,band_LOW_PAIRED
        + X__b7,r7,band_LOW_PAIRED + X__b8,r7,band_LOW_PAIRED
        + X__b5,r7,band_LOW_PAIRED + X__b3,r7,band_LOW_PAIRED
        + X__b1,r7,band_LOW_PAIRED + X__b0,r7,band_LOW_PAIRED <= 2
 c3741: - aux_cap_helper__b4,r10,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b4,r10,band_HIGH_PAIRED__condX1 <= 0
 c3742: - aux_cap_helper__b1,r3,band_LOW_PAIRED__condX1 <= 0
 c3743: - aux_cap_helper__b6,r1,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b6,r1,band_HIGH_PAIRED__condX1 <= 0
 c3744: - X__b7,r11,band_UNPAIRED
        + aux_cap_helper__b7,r11,band_UNPAIRED__condX1
        + aux_cap_helper__b7,r11,band_UNPAIRED__condX2  = 0
 c3745: aux_cap_helper__b2,r12,band_UNPAIRED__condX1
        - aux_cap_helper__b2,r12,band_UNPAIRED__Z1 <= 0
 c3746: aux_cap_helper__b6,r2,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b6,r2,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b6,r2,band_HIGH_PAIRED__Z1  = 1
 c3747: aux_cap_helper__b8,r6,band_UNPAIRED__condX1
        - aux_cap_helper__b8,r6,band_UNPAIRED__Z1 <= 0
 c3748: - 0.6181448971 aux_cap_helper__b1,r4,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b1,r4,band_HIGH_PAIRED__Z3
        - aux_cap_b1,r4,band_HIGH_PAIRED
        + 2.1635071398 aux_cap_helper__b1,r4,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b1,r4,band_HIGH_PAIRED__condX1
        + 1.3908260184 aux_cap_helper__b1,r4,band_HIGH_PAIRED__condX3  = 0
 c3749: - X__b1,r3,band_LOW_PAIRED
        + aux_cap_helper__b1,r3,band_LOW_PAIRED__condX2
        + aux_cap_helper__b1,r3,band_LOW_PAIRED__condX1  = 0
 c3750: - sv_function_helpervar_b6,r1__condX3
        + 5.8772093296743 sv_function_helpervar_b6,r1__Z3 <= 0
 c3751: aux_cap_helper__b8,r8,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b8,r8,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b8,r8,band_HIGH_PAIRED__Z2  = 1
 c3752: - 2 aux_cap_helper__b0,r4,band_LOW_PAIRED__Z2
        + aux_cap_helper__b0,r4,band_LOW_PAIRED__condX2 <= 0
 c3753: - aux_cap_b6,r5,band_HIGH_PAIRED
        - 0.6181448971 aux_cap_helper__b6,r5,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b6,r5,band_HIGH_PAIRED__Z3
        + 2.1635071398 aux_cap_helper__b6,r5,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b6,r5,band_HIGH_PAIRED__condX1
        + 1.3908260184 aux_cap_helper__b6,r5,band_HIGH_PAIRED__condX3  = 0
 c3754: - aux_sv_b2,r7 + 24.4944861086966 sv_function_helpervar_b2,r7__Z2
        + 54.6954001828529 sv_function_helpervar_b2,r7__Z3
        + 2.5373047613 sv_function_helpervar_b2,r7__condX3
        + 7.6947251993 sv_function_helpervar_b2,r7__condX2
        + 160.8367693688 sv_function_helpervar_b2,r7__condX1  = 0
 c3755: 141.882324615401 aux_sv_b8,r0 - aux_Omega_b8,r0  = 0
 c3756: 166.184186405236 aux_sv_b0,r0 - aux_Omega_b0,r0  = 0
 c3757: - aux_cap_helper__b0,r10,band_LOW_PAIRED__Z1
        + aux_cap_helper__b0,r10,band_LOW_PAIRED__condX1 <= 0
 c3758: - 15.9946187432626 sv_function_helpervar_b0,r0__Z3
        + sv_function_helpervar_b0,r0__condX3 <= 0
 c3759: aux_cap_helper__b3,r7,band_UNPAIRED__condX1
        - aux_cap_helper__b3,r7,band_UNPAIRED__Z1 <= 0
 c3760: aux_cap_helper__b3,r5,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b3,r5,band_HIGH_PAIRED__Z1 <= 0
 c3761: - aux_cap_helper__b6,r12,band_LOW_PAIRED__Z1
        + aux_cap_helper__b6,r12,band_LOW_PAIRED__condX1 <= 0
 c3762: - 0.3214355447 aux_cap_helper__b2,r10,band_UNPAIRED__Z2
        + 1.1250244065 aux_cap_helper__b2,r10,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b2,r10,band_UNPAIRED__condX1
        - aux_cap_b2,r10,band_UNPAIRED  = 0
 c3763: 621.463327233055 aux_sv_b5,r12 - aux_Omega_b5,r12  = 0
 c3764: sv_function_helpervar_b6,r7__condX2
        - 6.16442568412561 sv_function_helpervar_b6,r7__Z2 <= 0
 c3765: - 3 aux_cap_helper__b5,r4,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b5,r4,band_HIGH_PAIRED__condX3 <= 0
 c3766: - aux_sv_b5,r5 + 19.2290304415 sv_function_helpervar_b5,r5__condX3
        + 52.1971151664 sv_function_helpervar_b5,r5__condX2
        + 1169.5974424107 sv_function_helpervar_b5,r5__condX1
        + 178.723922182894 sv_function_helpervar_b5,r5__Z2
        + 385.300662430485 sv_function_helpervar_b5,r5__Z3  = 0
 c3767: - aux_cap_helper__b6,r8,band_LOW_PAIRED__condX2
        + aux_cap_helper__b6,r8,band_LOW_PAIRED__Z2 <= 0
 c3768: - sv_function_helpervar_b5,r9__condX2
        + 0.15994618751673 sv_function_helpervar_b5,r9__Z2 <= 0
 c3769: aux_cap_helper__b1,r5,band_UNPAIRED__Z2
        - aux_cap_helper__b1,r5,band_UNPAIRED__condX2 <= 0
 c3770: aux_cap_helper__b7,r1,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b7,r1,band_UNPAIRED__Z2 <= 0
 c3771: aux_cap_helper__b2,r2,band_UNPAIRED__condX1
        - aux_cap_helper__b2,r2,band_UNPAIRED__Z1 <= 0
 c3772: 0.927217345653617 aux_cap_helper__b2,r12,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b2,r12,band_HIGH_PAIRED__Z2
        + 1.5453622427 aux_cap_helper__b2,r12,band_HIGH_PAIRED__condX1
        + 2.1635071398 aux_cap_helper__b2,r12,band_HIGH_PAIRED__condX2
        + 1.3908260184 aux_cap_helper__b2,r12,band_HIGH_PAIRED__condX3
        - aux_cap_b2,r12,band_HIGH_PAIRED  = 0
 c3773: 1.3908260184 aux_cap_helper__b7,r13,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b7,r13,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b7,r13,band_HIGH_PAIRED__condX1
        - 0.6181448971 aux_cap_helper__b7,r13,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b7,r13,band_HIGH_PAIRED__Z3
        - aux_cap_b7,r13,band_HIGH_PAIRED  = 0
 c3774: - aux_cap_helper__b7,r7,band_LOW_PAIRED__Z1
        + aux_cap_helper__b7,r7,band_LOW_PAIRED__condX1 <= 0
 c3775: - aux_cap_b2,r4,band_HIGH_PAIRED
        + 1.5453622427 aux_cap_helper__b2,r4,band_HIGH_PAIRED__condX1
        + 2.1635071398 aux_cap_helper__b2,r4,band_HIGH_PAIRED__condX2
        + 1.3908260184 aux_cap_helper__b2,r4,band_HIGH_PAIRED__condX3
        - 0.6181448971 aux_cap_helper__b2,r4,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b2,r4,band_HIGH_PAIRED__Z3  = 0
 c3776: - aux_cap_helper__b3,r1,band_LOW_PAIRED__condX1 <= 0
 c3777: aux_cap_helper__b5,r7,band_LOW_PAIRED__condX1
        + aux_cap_helper__b5,r7,band_LOW_PAIRED__condX2
        - X__b5,r7,band_LOW_PAIRED  = 0
 c3778: 18497526.8940961 GlobalBidder_hat{W}__i8,k1 - GlobalBidder_PSI__i8,k1
        + aux_Omega_b8,r12 + aux_Omega_b8,r13 + aux_Omega_b8,r10
        + aux_Omega_b8,r11 + aux_Omega_b8,r5 + aux_Omega_b8,r6
        + aux_Omega_b8,r7 + aux_Omega_b8,r8 + aux_Omega_b8,r9 + aux_Omega_b8,r0
        + aux_Omega_b8,r1 + aux_Omega_b8,r2 + aux_Omega_b8,r3 + aux_Omega_b8,r4
        <= 18497526.8940961
 c3779: 1.5453622427 aux_cap_helper__b7,r7,band_HIGH_PAIRED__condX1
        + 2.1635071398 aux_cap_helper__b7,r7,band_HIGH_PAIRED__condX2
        + 1.3908260184 aux_cap_helper__b7,r7,band_HIGH_PAIRED__condX3
        - aux_cap_b7,r7,band_HIGH_PAIRED
        + 0.927217345653617 aux_cap_helper__b7,r7,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b7,r7,band_HIGH_PAIRED__Z2  = 0
 c3780: - aux_cap_b0,r3,band_HIGH_PAIRED
        + 1.3908260184 aux_cap_helper__b0,r3,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b0,r3,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b0,r3,band_HIGH_PAIRED__condX1
        + 0.927217345653617 aux_cap_helper__b0,r3,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b0,r3,band_HIGH_PAIRED__Z2  = 0
 c3781: - sv_function_helpervar_b8,r9__condX1 <= 0
 c3782: aux_cap_helper__b3,r11,band_LOW_PAIRED__Z2
        - aux_cap_helper__b3,r11,band_LOW_PAIRED__condX2 <= 0
 c3783: aux_cap_helper__b8,r1,band_UNPAIRED__condX1
        - aux_cap_helper__b8,r1,band_UNPAIRED__Z1 <= 0
 c3784: sv_function_helpervar_b4,r8__Z2 + sv_function_helpervar_b4,r8__Z1
        + sv_function_helpervar_b4,r8__Z3  = 1
 c3785: aux_cap_b9,r6,band_HIGH_PAIRED + aux_cap_b9,r6,band_LOW_PAIRED
        + aux_cap_b9,r6,band_UNPAIRED - aux_c_b9,r6  = 0
 c3786: aux_cap_helper__b9,r5,band_LOW_PAIRED__condX1
        - aux_cap_helper__b9,r5,band_LOW_PAIRED__Z1 <= 0
 c3787: aux_cap_helper__b7,r5,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b7,r5,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b7,r5,band_HIGH_PAIRED__Z2  = 1
 c3788: sv_function_helpervar_b7,r3__condX3
        - 15.9946187432626 sv_function_helpervar_b7,r3__Z3 <= 0
 c3789: - aux_sv_b2,r8 + 160.836769397 sv_function_helpervar_b2,r8__condX1
        + 6.8708387938 sv_function_helpervar_b2,r8__condX2
        + 2.7205749649 sv_function_helpervar_b2,r8__condX3
        + 24.6262635983739 sv_function_helpervar_b2,r8__Z2
        + 51.7640631487804 sv_function_helpervar_b2,r8__Z3  = 0
 c3790: - 0.159946187528613 sv_function_helpervar_b1,r5__Z1
        + sv_function_helpervar_b1,r5__condX1 <= 0
 c3791: aux_cap_helper__b1,r13,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b1,r13,band_UNPAIRED__Z2 <= 0
 c3792: - aux_cap_helper__b3,r11,band_LOW_PAIRED__condX1 <= 0
 c3793: - 2 aux_cap_helper__b3,r4,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b3,r4,band_HIGH_PAIRED__condX2 <= 0
 c3794: aux_cap_helper__b6,r11,band_UNPAIRED__Z1
        + aux_cap_helper__b6,r11,band_UNPAIRED__Z2  = 1
 c3795: - aux_cap_b6,r7,band_HIGH_PAIRED
        + 0.927217345653617 aux_cap_helper__b6,r7,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b6,r7,band_HIGH_PAIRED__Z2
        + 1.5453622427 aux_cap_helper__b6,r7,band_HIGH_PAIRED__condX1
        + 2.1635071398 aux_cap_helper__b6,r7,band_HIGH_PAIRED__condX2
        + 1.3908260184 aux_cap_helper__b6,r7,band_HIGH_PAIRED__condX3  = 0
 c3796: - aux_cap_helper__b7,r10,band_HIGH_PAIRED__condX1 <= 0
 c3797: 0.927217345653617 aux_cap_helper__b5,r4,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b5,r4,band_HIGH_PAIRED__Z2
        + 1.5453622427 aux_cap_helper__b5,r4,band_HIGH_PAIRED__condX1
        + 2.1635071398 aux_cap_helper__b5,r4,band_HIGH_PAIRED__condX2
        + 1.3908260184 aux_cap_helper__b5,r4,band_HIGH_PAIRED__condX3
        - aux_cap_b5,r4,band_HIGH_PAIRED  = 0
 c3798: - aux_cap_helper__b0,r9,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b0,r9,band_HIGH_PAIRED__condX1 <= 0
 c3799: 2 aux_cap_helper__b7,r5,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b7,r5,band_HIGH_PAIRED__condX3 <= 0
 c3800: sv_function_helpervar_b6,r0__Z1 + sv_function_helpervar_b6,r0__Z2
        + sv_function_helpervar_b6,r0__Z3  = 1
 c3801: - aux_cap_helper__b2,r1,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b2,r1,band_HIGH_PAIRED__Z2 <= 0
 c3802: aux_cap_helper__b3,r8,band_LOW_PAIRED__Z1
        + aux_cap_helper__b3,r8,band_LOW_PAIRED__Z2  = 1
 c3803: - 0.3214355447 aux_cap_helper__b0,r5,band_UNPAIRED__Z2
        + 0.8035888618 aux_cap_helper__b0,r5,band_UNPAIRED__condX1
        + 1.1250244065 aux_cap_helper__b0,r5,band_UNPAIRED__condX2
        - aux_cap_b0,r5,band_UNPAIRED  = 0
 c3804: aux_cap_helper__b4,r3,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b4,r3,band_LOW_PAIRED__Z2 <= 0
 c3805: - 0.3214355447 aux_cap_helper__b6,r9,band_UNPAIRED__Z2
        + 0.8035888618 aux_cap_helper__b6,r9,band_UNPAIRED__condX1
        + 1.1250244065 aux_cap_helper__b6,r9,band_UNPAIRED__condX2
        - aux_cap_b6,r9,band_UNPAIRED  = 0
 c3806: aux_cap_helper__b3,r9,band_UNPAIRED__Z1
        + aux_cap_helper__b3,r9,band_UNPAIRED__Z2  = 1
 c3807: - aux_cap_helper__b4,r7,band_LOW_PAIRED__condX1 <= 0
 c3808: 18.9591265255 sv_function_helpervar_b5,r9__condX3
        + 53.408540404 sv_function_helpervar_b5,r9__condX2
        + 1169.5974418232 sv_function_helpervar_b5,r9__condX1 - aux_sv_b5,r9
        + 178.530159330489 sv_function_helpervar_b5,r9__Z2
        + 389.617672664692 sv_function_helpervar_b5,r9__Z3  = 0
 c3809: - 0.159946187471719 sv_function_helpervar_b2,r13__Z1
        + sv_function_helpervar_b2,r13__condX1 <= 0
 c3810: aux_cap_helper__b7,r10,band_LOW_PAIRED__Z1
        + aux_cap_helper__b7,r10,band_LOW_PAIRED__Z2  = 1
 c3811: aux_cap_helper__b5,r13,band_UNPAIRED__Z1
        + aux_cap_helper__b5,r13,band_UNPAIRED__Z2  = 1
 c3812: aux_cap_helper__b8,r13,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b8,r13,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b8,r13,band_HIGH_PAIRED__Z3  = 1
 c3813: aux_cap_helper__b0,r11,band_LOW_PAIRED__Z1
        + aux_cap_helper__b0,r11,band_LOW_PAIRED__Z2  = 1
 c3814: - aux_cap_helper__b6,r4,band_UNPAIRED__condX1 <= 0
 c3815: - 0.159946187462914 sv_function_helpervar_b0,r13__Z1
        + sv_function_helpervar_b0,r13__condX1 <= 0
 c3816: 1.3908260184 aux_cap_helper__b1,r3,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b1,r3,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b1,r3,band_HIGH_PAIRED__condX1
        - aux_cap_b1,r3,band_HIGH_PAIRED
        - 0.6181448971 aux_cap_helper__b1,r3,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b1,r3,band_HIGH_PAIRED__Z3  = 0
 c3817: - sv_function_helpervar_b8,r2__condX3
        + 8.31036492628082 sv_function_helpervar_b8,r2__Z3 <= 0
 c3818: - aux_cap_helper__b8,r13,band_UNPAIRED__condX1 <= 0
 c3819: - 0.6181448971 aux_cap_helper__b8,r5,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b8,r5,band_HIGH_PAIRED__Z3
        + 2.1635071398 aux_cap_helper__b8,r5,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b8,r5,band_HIGH_PAIRED__condX1
        + 1.3908260184 aux_cap_helper__b8,r5,band_HIGH_PAIRED__condX3
        - aux_cap_b8,r5,band_HIGH_PAIRED  = 0
 c3820: - aux_cap_helper__b3,r1,band_LOW_PAIRED__condX2
        + aux_cap_helper__b3,r1,band_LOW_PAIRED__Z2 <= 0
 c3821: - aux_cap_helper__b9,r9,band_HIGH_PAIRED__condX1 <= 0
 c3822: aux_cap_helper__b3,r8,band_LOW_PAIRED__condX1
        - aux_cap_helper__b3,r8,band_LOW_PAIRED__Z1 <= 0
 c3823: sv_function_helpervar_b4,r8__condX3
        + sv_function_helpervar_b4,r8__condX2
        + sv_function_helpervar_b4,r8__condX1 - aux_c_b4,r8  = 0
 c3824: aux_cap_helper__b0,r5,band_LOW_PAIRED__Z1
        + aux_cap_helper__b0,r5,band_LOW_PAIRED__Z2  = 1
 c3825: - aux_Omega_b3,r12 + 820.722531034619 aux_sv_b3,r12  = 0
 c3826: aux_cap_b9,r10,band_HIGH_PAIRED + aux_cap_b9,r10,band_UNPAIRED
        - aux_c_b9,r10 + aux_cap_b9,r10,band_LOW_PAIRED  = 0
 c3827: 0.927217345653617 aux_cap_helper__b4,r4,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b4,r4,band_HIGH_PAIRED__Z2
        + 1.5453622427 aux_cap_helper__b4,r4,band_HIGH_PAIRED__condX1
        + 2.1635071398 aux_cap_helper__b4,r4,band_HIGH_PAIRED__condX2
        + 1.3908260184 aux_cap_helper__b4,r4,band_HIGH_PAIRED__condX3
        - aux_cap_b4,r4,band_HIGH_PAIRED  = 0
 c3828: - sv_function_helpervar_b9,r6__condX2
        + 0.159946187446589 sv_function_helpervar_b9,r6__Z2 <= 0
 c3829: aux_cap_helper__b4,r11,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b4,r11,band_HIGH_PAIRED__Z2 <= 0
 c3830: - aux_cap_helper__b5,r9,band_LOW_PAIRED__Z1
        + aux_cap_helper__b5,r9,band_LOW_PAIRED__condX1 <= 0
 c3831: 0.8035888618 aux_cap_helper__b2,r2,band_UNPAIRED__condX1
        + 1.1250244065 aux_cap_helper__b2,r2,band_UNPAIRED__condX2
        - 0.3214355447 aux_cap_helper__b2,r2,band_UNPAIRED__Z2
        - aux_cap_b2,r2,band_UNPAIRED  = 0
 c3832: 1.1250244065 aux_cap_helper__b8,r6,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b8,r6,band_UNPAIRED__condX1
        - aux_cap_b8,r6,band_UNPAIRED
        - 0.3214355447 aux_cap_helper__b8,r6,band_UNPAIRED__Z2  = 0
 c3833: - sv_function_helpervar_b6,r12__condX1 <= 0
 c3834: - aux_cap_helper__b0,r0,band_UNPAIRED__condX1 <= 0
 c3835: aux_cap_helper__b2,r2,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b2,r2,band_LOW_PAIRED__Z2 <= 0
 c3836: sv_function_helpervar_b1,r12__condX3
        - 15.9946187432626 sv_function_helpervar_b1,r12__Z3 <= 0
 c3837: - aux_cap_helper__b4,r8,band_HIGH_PAIRED__condX1 <= 0
 c3838: 6.124193939318 sv_function_helpervar_b2,r12__Z3
        - sv_function_helpervar_b2,r12__condX3 <= 0
 c3839: sv_function_helpervar_b3,r13__condX3
        + sv_function_helpervar_b3,r13__condX2
        + sv_function_helpervar_b3,r13__condX1 - aux_c_b3,r13  = 0
 c3840: - sv_function_helpervar_b9,r7__condX1 <= 0
 c3841: - aux_cap_b3,r9,band_UNPAIRED
        + 1.1250244065 aux_cap_helper__b3,r9,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b3,r9,band_UNPAIRED__condX1
        - 0.3214355447 aux_cap_helper__b3,r9,band_UNPAIRED__Z2  = 0
 c3842: - aux_cap_b4,r2,band_LOW_PAIRED
        - 1.4943850124 aux_cap_helper__b4,r2,band_LOW_PAIRED__Z2
        + 5.2303475434 aux_cap_helper__b4,r2,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b4,r2,band_LOW_PAIRED__condX1  = 0
 c3843: sv_function_helpervar_b4,r0__Z3 + sv_function_helpervar_b4,r0__Z1
        + sv_function_helpervar_b4,r0__Z2  = 1
 c3844: - sv_function_helpervar_b7,r6__condX2
        + 0.159946187459757 sv_function_helpervar_b7,r6__Z2 <= 0
 c3845: - aux_cap_helper__b4,r10,band_LOW_PAIRED__condX1 <= 0
 c3846: 2.1635071398 aux_cap_helper__b7,r8,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b7,r8,band_HIGH_PAIRED__condX1
        + 1.3908260184 aux_cap_helper__b7,r8,band_HIGH_PAIRED__condX3
        - 0.6181448971 aux_cap_helper__b7,r8,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b7,r8,band_HIGH_PAIRED__Z3
        - aux_cap_b7,r8,band_HIGH_PAIRED  = 0
 c3847: aux_cap_helper__b6,r4,band_UNPAIRED__Z2
        - aux_cap_helper__b6,r4,band_UNPAIRED__condX2 <= 0
 c3848: - aux_sv_b8,r4 + 27.7042928264 sv_function_helpervar_b8,r4__condX3
        + 1343.0085751022 sv_function_helpervar_b8,r4__condX1
        + 45.2876584705 sv_function_helpervar_b8,r4__condX2
        + 352.469662959521 sv_function_helpervar_b8,r4__Z3
        + 207.565513039889 sv_function_helpervar_b8,r4__Z2  = 0
 c3849: - aux_Omega_b2,r3 + 454.669171309364 aux_sv_b2,r3  = 0
 c3850: - sv_function_helpervar_b5,r4__condX1 <= 0
 c3851: - aux_cap_helper__b8,r13,band_UNPAIRED__condX2
        + aux_cap_helper__b8,r13,band_UNPAIRED__Z2 <= 0
 c3852: aux_cap_helper__b1,r3,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b1,r3,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b1,r3,band_HIGH_PAIRED__Z1  = 1
 c3853: - 1.4943850124 aux_cap_helper__b2,r1,band_LOW_PAIRED__Z2
        - aux_cap_b2,r1,band_LOW_PAIRED
        + 5.2303475434 aux_cap_helper__b2,r1,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b2,r1,band_LOW_PAIRED__condX1  = 0
 c3854: 6.18724675079566 sv_function_helpervar_b3,r11__Z3
        - sv_function_helpervar_b3,r11__condX3 <= 0
 c3855: - sv_function_helpervar_b4,r6__condX1 <= 0
 c3856: - 3 aux_cap_helper__b0,r12,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b0,r12,band_HIGH_PAIRED__condX3 <= 0
 c3857: - aux_cap_helper__b2,r1,band_HIGH_PAIRED__condX1 <= 0
 c3858: - 15.9946187432626 sv_function_helpervar_b6,r5__Z3
        + sv_function_helpervar_b6,r5__condX3 <= 0
 c3859: 1.5453622427 aux_cap_helper__b3,r7,band_HIGH_PAIRED__condX1
        + 2.1635071398 aux_cap_helper__b3,r7,band_HIGH_PAIRED__condX2
        + 1.3908260184 aux_cap_helper__b3,r7,band_HIGH_PAIRED__condX3
        - aux_cap_b3,r7,band_HIGH_PAIRED
        + 0.927217345653617 aux_cap_helper__b3,r7,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b3,r7,band_HIGH_PAIRED__Z2  = 0
 c3860: aux_cap_b6,r5,band_HIGH_PAIRED - aux_c_b6,r5
        + aux_cap_b6,r5,band_UNPAIRED + aux_cap_b6,r5,band_LOW_PAIRED  = 0
 c3861: - aux_cap_helper__b8,r6,band_LOW_PAIRED__condX1 <= 0
 c3862: - aux_cap_helper__b4,r9,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b4,r9,band_HIGH_PAIRED__Z3 <= 0
 c3863: aux_cap_helper__b4,r6,band_UNPAIRED__condX2
        + aux_cap_helper__b4,r6,band_UNPAIRED__condX1 - X__b4,r6,band_UNPAIRED
         = 0
 c3864: aux_cap_helper__b5,r13,band_LOW_PAIRED__condX1
        - aux_cap_helper__b5,r13,band_LOW_PAIRED__Z1 <= 0
 c3865: sv_function_helpervar_b7,r9__Z1 + sv_function_helpervar_b7,r9__Z2
        + sv_function_helpervar_b7,r9__Z3  = 1
 c3866: aux_cap_helper__b1,r1,band_LOW_PAIRED__Z1
        + aux_cap_helper__b1,r1,band_LOW_PAIRED__Z2  = 1
 c3867: aux_cap_helper__b6,r0,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b6,r0,band_HIGH_PAIRED__Z2 <= 0
 c3868: - aux_cap_helper__b3,r6,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b3,r6,band_HIGH_PAIRED__condX1 <= 0
 c3869: - aux_cap_helper__b4,r7,band_UNPAIRED__Z1
        + aux_cap_helper__b4,r7,band_UNPAIRED__condX1 <= 0
 c3870: X__b0,r12,band_UNPAIRED + X__b8,r12,band_UNPAIRED
        + X__b6,r12,band_UNPAIRED + X__b3,r12,band_UNPAIRED
        + X__b9,r12,band_UNPAIRED + X__b2,r12,band_UNPAIRED
        + X__b7,r12,band_UNPAIRED + X__b4,r12,band_UNPAIRED
        + X__b5,r12,band_UNPAIRED + X__b1,r12,band_UNPAIRED <= 2
 c3871: aux_cap_helper__b0,r7,band_LOW_PAIRED__condX1
        - aux_cap_helper__b0,r7,band_LOW_PAIRED__Z1 <= 0
 c3872: aux_cap_helper__b7,r6,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b7,r6,band_UNPAIRED__Z2 <= 0
 c3873: - aux_cap_b6,r10,band_LOW_PAIRED
        - 1.4943850124 aux_cap_helper__b6,r10,band_LOW_PAIRED__Z2
        + 3.735962531 aux_cap_helper__b6,r10,band_LOW_PAIRED__condX1
        + 5.2303475434 aux_cap_helper__b6,r10,band_LOW_PAIRED__condX2  = 0
 c3874: 6.07097130497179 sv_function_helpervar_b1,r13__Z3
        - sv_function_helpervar_b1,r13__condX3 <= 0
 c3875: 0.159946187476732 sv_function_helpervar_b8,r1__Z2
        - sv_function_helpervar_b8,r1__condX2 <= 0
 c3876: - 3 aux_cap_helper__b7,r9,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b7,r9,band_HIGH_PAIRED__condX3 <= 0
 c3877: aux_cap_helper__b4,r10,band_LOW_PAIRED__Z2
        - aux_cap_helper__b4,r10,band_LOW_PAIRED__condX2 <= 0
 c3878: aux_cap_b4,r13,band_UNPAIRED - aux_c_b4,r13
        + aux_cap_b4,r13,band_LOW_PAIRED + aux_cap_b4,r13,band_HIGH_PAIRED  = 0
 c3879: - aux_cap_helper__b7,r10,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b7,r10,band_HIGH_PAIRED__Z2 <= 0
 c3880: aux_cap_helper__b5,r0,band_LOW_PAIRED__condX1
        + aux_cap_helper__b5,r0,band_LOW_PAIRED__condX2
        - X__b5,r0,band_LOW_PAIRED  = 0
 c3881: 0.159946187488921 sv_function_helpervar_b2,r11__Z2
        - sv_function_helpervar_b2,r11__condX2 <= 0
 c3882: 6.9785169243679 sv_function_helpervar_b9,r1__Z3
        - sv_function_helpervar_b9,r1__condX3 <= 0
 c3883: sv_function_helpervar_b9,r5__condX3
        - 15.9946187432626 sv_function_helpervar_b9,r5__Z3 <= 0
 c3884: aux_cap_helper__b0,r12,band_UNPAIRED__Z2
        + aux_cap_helper__b0,r12,band_UNPAIRED__Z1  = 1
 c3885: 0.927217345653617 aux_cap_helper__b1,r7,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b1,r7,band_HIGH_PAIRED__Z2
        - aux_cap_b1,r7,band_HIGH_PAIRED
        + 1.3908260184 aux_cap_helper__b1,r7,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b1,r7,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b1,r7,band_HIGH_PAIRED__condX1  = 0
 c3886: aux_cap_helper__b2,r5,band_LOW_PAIRED__Z2
        + aux_cap_helper__b2,r5,band_LOW_PAIRED__Z1  = 1
 c3887: - 0.159946187436519 sv_function_helpervar_b6,r4__Z1
        + sv_function_helpervar_b6,r4__condX1 <= 0
 c3888: aux_cap_helper__b0,r0,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b0,r0,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b0,r0,band_HIGH_PAIRED__condX1
        - X__b0,r0,band_HIGH_PAIRED  = 0
 c3889: aux_cap_helper__b4,r12,band_UNPAIRED__Z2
        - aux_cap_helper__b4,r12,band_UNPAIRED__condX2 <= 0
 c3890: 0.927217345653617 aux_cap_helper__b5,r8,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b5,r8,band_HIGH_PAIRED__Z2
        + 1.3908260184 aux_cap_helper__b5,r8,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b5,r8,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b5,r8,band_HIGH_PAIRED__condX1
        - aux_cap_b5,r8,band_HIGH_PAIRED  = 0
 c3891: 0.159946187520236 sv_function_helpervar_b7,r0__Z2
        - sv_function_helpervar_b7,r0__condX2 <= 0
 c3892: - X__b6,r2,band_LOW_PAIRED
        + aux_cap_helper__b6,r2,band_LOW_PAIRED__condX2
        + aux_cap_helper__b6,r2,band_LOW_PAIRED__condX1  = 0
 c3893: - 2 aux_cap_helper__b5,r9,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b5,r9,band_HIGH_PAIRED__condX2 <= 0
 c3894: - X__b1,r2,band_HIGH_PAIRED
        + aux_cap_helper__b1,r2,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b1,r2,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b1,r2,band_HIGH_PAIRED__condX2  = 0
 c3895: aux_cap_helper__b5,r8,band_LOW_PAIRED__Z2
        + aux_cap_helper__b5,r8,band_LOW_PAIRED__Z1  = 1
 c3896: sv_function_helpervar_b9,r3__condX1
        - 0.159946187444714 sv_function_helpervar_b9,r3__Z1 <= 0
 c3897: - X__b4,r12,band_LOW_PAIRED
        + aux_cap_helper__b4,r12,band_LOW_PAIRED__condX2
        + aux_cap_helper__b4,r12,band_LOW_PAIRED__condX1  = 0
 c3898: - aux_cap_b9,r4,band_HIGH_PAIRED
        + 1.5453622427 aux_cap_helper__b9,r4,band_HIGH_PAIRED__condX1
        + 2.1635071398 aux_cap_helper__b9,r4,band_HIGH_PAIRED__condX2
        + 1.3908260184 aux_cap_helper__b9,r4,band_HIGH_PAIRED__condX3
        - 0.6181448971 aux_cap_helper__b9,r4,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b9,r4,band_HIGH_PAIRED__Z3  = 0
 c3899: sv_function_helpervar_b6,r5__Z3 + sv_function_helpervar_b6,r5__Z2
        + sv_function_helpervar_b6,r5__Z1  = 1
 c3900: - aux_cap_helper__b8,r3,band_LOW_PAIRED__condX2
        + aux_cap_helper__b8,r3,band_LOW_PAIRED__Z2 <= 0
 c3901: - aux_cap_helper__b1,r0,band_UNPAIRED__condX1 <= 0
 c3902: sv_function_helpervar_b9,r7__condX3
        + sv_function_helpervar_b9,r7__condX2
        + sv_function_helpervar_b9,r7__condX1 - aux_c_b9,r7  = 0
 c3903: - aux_cap_helper__b8,r3,band_LOW_PAIRED__condX1 <= 0
 c3904: X__b8,r4,band_HIGH_PAIRED + X__b4,r4,band_HIGH_PAIRED
        + X__b7,r4,band_HIGH_PAIRED + X__b9,r4,band_HIGH_PAIRED
        + X__b6,r4,band_HIGH_PAIRED + X__b5,r4,band_HIGH_PAIRED
        + X__b0,r4,band_HIGH_PAIRED + X__b1,r4,band_HIGH_PAIRED
        + X__b3,r4,band_HIGH_PAIRED + X__b2,r4,band_HIGH_PAIRED <= 3
 c3905: - X__b2,r5,band_LOW_PAIRED
        + aux_cap_helper__b2,r5,band_LOW_PAIRED__condX1
        + aux_cap_helper__b2,r5,band_LOW_PAIRED__condX2  = 0
 c3906: 3.735962531 aux_cap_helper__b4,r9,band_LOW_PAIRED__condX1
        + 5.2303475434 aux_cap_helper__b4,r9,band_LOW_PAIRED__condX2
        - aux_cap_b4,r9,band_LOW_PAIRED
        - 1.4943850124 aux_cap_helper__b4,r9,band_LOW_PAIRED__Z2  = 0
 c3907: - aux_cap_helper__b9,r8,band_LOW_PAIRED__Z1
        + aux_cap_helper__b9,r8,band_LOW_PAIRED__condX1 <= 0
 c3908: 2 aux_cap_helper__b5,r0,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b5,r0,band_HIGH_PAIRED__condX3 <= 0
 c3909: sv_function_helpervar_b6,r8__Z2 + sv_function_helpervar_b6,r8__Z1
        + sv_function_helpervar_b6,r8__Z3  = 1
 c3910: aux_cap_helper__b1,r6,band_UNPAIRED__Z1
        + aux_cap_helper__b1,r6,band_UNPAIRED__Z2  = 1
 c3911: GlobalBidder_W_ir__b9,r11 + GlobalBidder_W_ir__b9,r12
        + GlobalBidder_W_ir__b9,r10 + GlobalBidder_W_ir__b9,r13
        + GlobalBidder_W_ir__b9,r6 + GlobalBidder_W_ir__b9,r7
        + GlobalBidder_W_ir__b9,r8 + GlobalBidder_W_ir__b9,r9
        - GlobalBidder_W_i_9 + GlobalBidder_W_ir__b9,r2
        + GlobalBidder_W_ir__b9,r3 + GlobalBidder_W_ir__b9,r4
        + GlobalBidder_W_ir__b9,r5 + GlobalBidder_W_ir__b9,r0
        + GlobalBidder_W_ir__b9,r1  = 0
 c3912: - 6.19171029678876 sv_function_helpervar_b2,r11__Z2
        + sv_function_helpervar_b2,r11__condX2 <= 0
 c3913: aux_cap_helper__b6,r11,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b6,r11,band_HIGH_PAIRED__Z3 <= 0
 c3914: 0.159946187451652 sv_function_helpervar_b5,r12__Z2
        - sv_function_helpervar_b5,r12__condX2 <= 0
 c3915: - aux_c_b7,r9 + aux_cap_b7,r9,band_HIGH_PAIRED
        + aux_cap_b7,r9,band_LOW_PAIRED + aux_cap_b7,r9,band_UNPAIRED  = 0
 c3916: - 15.9946187432626 sv_function_helpervar_b3,r0__Z3
        + sv_function_helpervar_b3,r0__condX3 <= 0
 c3917: 2 aux_cap_helper__b2,r0,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b2,r0,band_HIGH_PAIRED__condX3 <= 0
 c3918: - X__b0,r7,band_HIGH_PAIRED
        + aux_cap_helper__b0,r7,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b0,r7,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b0,r7,band_HIGH_PAIRED__condX1  = 0
 c3919: - 1.4943850124 aux_cap_helper__b5,r5,band_LOW_PAIRED__Z2
        + 5.2303475434 aux_cap_helper__b5,r5,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b5,r5,band_LOW_PAIRED__condX1
        - aux_cap_b5,r5,band_LOW_PAIRED  = 0
 c3920: aux_cap_b6,r7,band_HIGH_PAIRED + aux_cap_b6,r7,band_UNPAIRED
        - aux_c_b6,r7 + aux_cap_b6,r7,band_LOW_PAIRED  = 0
 c3921: 1.5453622427 aux_cap_helper__b0,r6,band_HIGH_PAIRED__condX1
        + 1.3908260184 aux_cap_helper__b0,r6,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b0,r6,band_HIGH_PAIRED__condX2
        - aux_cap_b0,r6,band_HIGH_PAIRED
        + 0.927217345653617 aux_cap_helper__b0,r6,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b0,r6,band_HIGH_PAIRED__Z2  = 0
 c3922: - aux_c_b2,r8 + sv_function_helpervar_b2,r8__condX1
        + sv_function_helpervar_b2,r8__condX2
        + sv_function_helpervar_b2,r8__condX3  = 0
 c3923: - X__b6,r0,band_LOW_PAIRED
        + aux_cap_helper__b6,r0,band_LOW_PAIRED__condX2
        + aux_cap_helper__b6,r0,band_LOW_PAIRED__condX1  = 0
 c3924: - aux_cap_b7,r10,band_HIGH_PAIRED
        + 1.5453622427 aux_cap_helper__b7,r10,band_HIGH_PAIRED__condX1
        + 1.3908260184 aux_cap_helper__b7,r10,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b7,r10,band_HIGH_PAIRED__condX2
        - 0.6181448971 aux_cap_helper__b7,r10,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b7,r10,band_HIGH_PAIRED__Z3  = 0
 c3925: - sv_function_helpervar_b8,r11__condX2
        + 0.159946187501739 sv_function_helpervar_b8,r11__Z2 <= 0
 c3926: - aux_cap_helper__b1,r0,band_UNPAIRED__condX2
        + aux_cap_helper__b1,r0,band_UNPAIRED__Z2 <= 0
 c3927: - 2 aux_cap_helper__b8,r2,band_LOW_PAIRED__Z2
        + aux_cap_helper__b8,r2,band_LOW_PAIRED__condX2 <= 0
 c3928: GlobalBidder_W_i_9 - 14 GlobalBidder_hat{W}__i9,k0 >= 0
 c3929: aux_cap_helper__b3,r9,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b3,r9,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b3,r9,band_HIGH_PAIRED__Z2  = 1
 c3930: aux_cap_helper__b5,r1,band_UNPAIRED__Z2
        + aux_cap_helper__b5,r1,band_UNPAIRED__Z1  = 1
 c3931: aux_cap_helper__b9,r1,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b9,r1,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b9,r1,band_HIGH_PAIRED__Z2  = 1
 c3932: - aux_cap_b5,r11,band_LOW_PAIRED
        - 1.4943850124 aux_cap_helper__b5,r11,band_LOW_PAIRED__Z2
        + 5.2303475434 aux_cap_helper__b5,r11,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b5,r11,band_LOW_PAIRED__condX1  = 0
 c3933: aux_cap_helper__b6,r2,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b6,r2,band_HIGH_PAIRED__Z1 <= 0
 c3934: - aux_cap_b7,r0,band_LOW_PAIRED
        - 1.4943850124 aux_cap_helper__b7,r0,band_LOW_PAIRED__Z2
        + 3.735962531 aux_cap_helper__b7,r0,band_LOW_PAIRED__condX1
        + 5.2303475434 aux_cap_helper__b7,r0,band_LOW_PAIRED__condX2  = 0
 c3935: sv_function_helpervar_b8,r13__Z1 + sv_function_helpervar_b8,r13__Z2
        + sv_function_helpervar_b8,r13__Z3  = 1
 c3936: - aux_cap_helper__b4,r12,band_UNPAIRED__condX1 <= 0
 c3937: aux_cap_helper__b5,r1,band_UNPAIRED__condX2
        + aux_cap_helper__b5,r1,band_UNPAIRED__condX1 - X__b5,r1,band_UNPAIRED
         = 0
 c3938: 1059.4718675177 sv_function_helpervar_b6,r12__condX1
        + 50.8615141712 sv_function_helpervar_b6,r12__condX2
        + 16.6817162441 sv_function_helpervar_b6,r12__condX3 - aux_sv_b6,r12
        + 360.806330590935 sv_function_helpervar_b6,r12__Z3
        + 161.323380633098 sv_function_helpervar_b6,r12__Z2  = 0
 c3939: - 5.79966606576978 sv_function_helpervar_b4,r6__Z2
        + sv_function_helpervar_b4,r6__condX2 <= 0
 c3940: - sv_function_helpervar_b2,r1__condX3
        + 5.90199764671298 sv_function_helpervar_b2,r1__Z3 <= 0
 c3941: 471.810931143106 sv_function_helpervar_b3,r4__Z3
        + 209.956598128917 sv_function_helpervar_b3,r4__Z2
        + 1379.5885615455 sv_function_helpervar_b3,r4__condX1
        + 21.5977684079 sv_function_helpervar_b3,r4__condX3
        + 66.9183350023 sv_function_helpervar_b3,r4__condX2 - aux_sv_b3,r4  = 0
 c3942: aux_cap_helper__b4,r5,band_LOW_PAIRED__condX2
        + aux_cap_helper__b4,r5,band_LOW_PAIRED__condX1
        - X__b4,r5,band_LOW_PAIRED  = 0
 c3943: aux_cap_helper__b0,r1,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b0,r1,band_LOW_PAIRED__Z2 <= 0
 c3944: - aux_c_b2,r12 + sv_function_helpervar_b2,r12__condX2
        + sv_function_helpervar_b2,r12__condX3
        + sv_function_helpervar_b2,r12__condX1  = 0
 c3945: - 0.159946187488557 sv_function_helpervar_b9,r2__Z1
        + sv_function_helpervar_b9,r2__condX1 <= 0
 c3946: 2.1635071398 aux_cap_helper__b2,r1,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b2,r1,band_HIGH_PAIRED__condX1
        + 1.3908260184 aux_cap_helper__b2,r1,band_HIGH_PAIRED__condX3
        - aux_cap_b2,r1,band_HIGH_PAIRED
        - 0.6181448971 aux_cap_helper__b2,r1,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b2,r1,band_HIGH_PAIRED__Z3  = 0
 c3947: - aux_c_b0,r0 + sv_function_helpervar_b0,r0__condX3
        + sv_function_helpervar_b0,r0__condX2
        + sv_function_helpervar_b0,r0__condX1  = 0
 c3948: sv_function_helpervar_b5,r4__condX2
        - 5.69598108125028 sv_function_helpervar_b5,r4__Z2 <= 0
 c3949: 0.0102040816326531 X__b8,r9,band_LOW_PAIRED
        + 0.0102040816326531 X__b8,r9,band_HIGH_PAIRED
        + 0.0102040816326531 X__b8,r9,band_UNPAIRED - GlobalBidder_W_ir__b8,r9
        <= 0
 c3950: aux_cap_helper__b1,r12,band_UNPAIRED__condX2
        + aux_cap_helper__b1,r12,band_UNPAIRED__condX1
        - X__b1,r12,band_UNPAIRED  = 0
 c3951: - 0.6181448971 aux_cap_helper__b4,r7,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b4,r7,band_HIGH_PAIRED__Z3
        + 1.3908260184 aux_cap_helper__b4,r7,band_HIGH_PAIRED__condX3
        + 1.5453622427 aux_cap_helper__b4,r7,band_HIGH_PAIRED__condX1
        + 2.1635071398 aux_cap_helper__b4,r7,band_HIGH_PAIRED__condX2
        - aux_cap_b4,r7,band_HIGH_PAIRED  = 0
 c3952: 0.8035888618 aux_cap_helper__b8,r8,band_UNPAIRED__condX1
        + 1.1250244065 aux_cap_helper__b8,r8,band_UNPAIRED__condX2
        - 0.3214355447 aux_cap_helper__b8,r8,band_UNPAIRED__Z2
        - aux_cap_b8,r8,band_UNPAIRED  = 0
 c3953: - sv_function_helpervar_b6,r7__condX1 <= 0
 c3954: aux_cap_helper__b4,r11,band_UNPAIRED__condX2
        + aux_cap_helper__b4,r11,band_UNPAIRED__condX1
        - X__b4,r11,band_UNPAIRED  = 0
 c3955: aux_cap_helper__b9,r6,band_UNPAIRED__Z2
        + aux_cap_helper__b9,r6,band_UNPAIRED__Z1  = 1
 c3956: aux_cap_b1,r5,band_UNPAIRED + aux_cap_b1,r5,band_LOW_PAIRED
        + aux_cap_b1,r5,band_HIGH_PAIRED - aux_c_b1,r5  = 0
 c3957: - sv_function_helpervar_b3,r10__condX2
        + 0.159946187535581 sv_function_helpervar_b3,r10__Z2 <= 0
 c3958: - 2 aux_cap_helper__b1,r1,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b1,r1,band_HIGH_PAIRED__condX2 <= 0
 c3959: 0.927217345653617 aux_cap_helper__b2,r11,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b2,r11,band_HIGH_PAIRED__Z2
        - aux_cap_b2,r11,band_HIGH_PAIRED
        + 1.3908260184 aux_cap_helper__b2,r11,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b2,r11,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b2,r11,band_HIGH_PAIRED__condX1  = 0
 c3960: - aux_sv_b2,r10 + 2.72419743 sv_function_helpervar_b2,r10__condX3
        + 6.8573219543 sv_function_helpervar_b2,r10__condX2
        + 160.8367694108 sv_function_helpervar_b2,r10__condX1
        + 51.7061232013866 sv_function_helpervar_b2,r10__Z3
        + 24.6284255654081 sv_function_helpervar_b2,r10__Z2  = 0
 c3961: - aux_cap_helper__b1,r13,band_LOW_PAIRED__condX1 <= 0
 c3962: aux_cap_helper__b0,r8,band_LOW_PAIRED__condX2
        + aux_cap_helper__b0,r8,band_LOW_PAIRED__condX1
        - X__b0,r8,band_LOW_PAIRED  = 0
 c3963: - GlobalBidder_W_ir__b9,r6 + X__b9,r6,band_UNPAIRED
        + X__b9,r6,band_HIGH_PAIRED + X__b9,r6,band_LOW_PAIRED >= 0
 c3964: aux_cap_helper__b4,r7,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b4,r7,band_HIGH_PAIRED__condX2 <= 0
 c3965: 2 aux_cap_helper__b2,r13,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b2,r13,band_HIGH_PAIRED__condX3 <= 0
 c3966: - aux_cap_helper__b7,r12,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b7,r12,band_HIGH_PAIRED__Z3 <= 0
 c3967: sv_function_helpervar_b7,r2__condX1
        + sv_function_helpervar_b7,r2__condX3
        + sv_function_helpervar_b7,r2__condX2 - aux_c_b7,r2  = 0
 c3968: 0.8035888618 aux_cap_helper__b3,r7,band_UNPAIRED__condX1
        + 1.1250244065 aux_cap_helper__b3,r7,band_UNPAIRED__condX2
        - 0.3214355447 aux_cap_helper__b3,r7,band_UNPAIRED__Z2
        - aux_cap_b3,r7,band_UNPAIRED  = 0
 c3969: aux_cap_helper__b5,r1,band_UNPAIRED__condX1
        - aux_cap_helper__b5,r1,band_UNPAIRED__Z1 <= 0
 c3970: aux_cap_helper__b6,r10,band_LOW_PAIRED__Z2
        + aux_cap_helper__b6,r10,band_LOW_PAIRED__Z1  = 1
 c3971: sv_function_helpervar_b6,r9__condX2
        - 6.20098161125853 sv_function_helpervar_b6,r9__Z2 <= 0
 c3972: sv_function_helpervar_b0,r2__condX1
        - 0.159946187506047 sv_function_helpervar_b0,r2__Z1 <= 0
 c3973: - 0.159946187436407 sv_function_helpervar_b8,r13__Z1
        + sv_function_helpervar_b8,r13__condX1 <= 0
 c3974: 282.899976487001 sv_function_helpervar_b7,r8__Z2
        + 605.883073694854 sv_function_helpervar_b7,r8__Z3 - aux_sv_b7,r8
        + 81.6036020112 sv_function_helpervar_b7,r8__condX2
        + 30.6500611785 sv_function_helpervar_b7,r8__condX3
        + 1850.3233259656 sv_function_helpervar_b7,r8__condX1  = 0
 c3975: - 2 aux_cap_helper__b4,r0,band_LOW_PAIRED__Z2
        + aux_cap_helper__b4,r0,band_LOW_PAIRED__condX2 <= 0
 c3976: - 6.6833780439137 sv_function_helpervar_b8,r9__Z2
        + sv_function_helpervar_b8,r9__condX2 <= 0
 c3977: aux_cap_helper__b8,r0,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b8,r0,band_HIGH_PAIRED__Z3 <= 0
 c3978: - sv_function_helpervar_b3,r9__condX2
        + 0.159946187458878 sv_function_helpervar_b3,r9__Z2 <= 0
 c3979: aux_cap_helper__b0,r2,band_UNPAIRED__condX1
        - aux_cap_helper__b0,r2,band_UNPAIRED__Z1 <= 0
 c3980: - sv_function_helpervar_b6,r13__condX2
        + 0.159946187451606 sv_function_helpervar_b6,r13__Z2 <= 0
 c3981: aux_cap_helper__b3,r7,band_LOW_PAIRED__Z2
        - aux_cap_helper__b3,r7,band_LOW_PAIRED__condX2 <= 0
 c3982: - aux_cap_helper__b5,r2,band_LOW_PAIRED__condX2
        + aux_cap_helper__b5,r2,band_LOW_PAIRED__Z2 <= 0
 c3983: - sv_function_helpervar_b7,r13__condX1 <= 0
 c3984: - aux_cap_helper__b1,r3,band_LOW_PAIRED__condX2
        + aux_cap_helper__b1,r3,band_LOW_PAIRED__Z2 <= 0
 c3985: - aux_cap_helper__b6,r8,band_LOW_PAIRED__condX1 <= 0
 c3986: - aux_cap_helper__b1,r5,band_UNPAIRED__condX1 <= 0
 c3987: - aux_cap_helper__b6,r7,band_UNPAIRED__Z1
        + aux_cap_helper__b6,r7,band_UNPAIRED__condX1 <= 0
 c3988: sv_function_helpervar_b0,r3__condX2
        + sv_function_helpervar_b0,r3__condX3
        + sv_function_helpervar_b0,r3__condX1 - aux_c_b0,r3  = 0
 c3989: - aux_cap_helper__b4,r13,band_LOW_PAIRED__condX2
        + aux_cap_helper__b4,r13,band_LOW_PAIRED__Z2 <= 0
 c3990: - aux_cap_helper__b5,r2,band_LOW_PAIRED__condX1 <= 0
 c3991: 1.3908260184 aux_cap_helper__b2,r3,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b2,r3,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b2,r3,band_HIGH_PAIRED__condX1
        - aux_cap_b2,r3,band_HIGH_PAIRED
        + 0.927217345653617 aux_cap_helper__b2,r3,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b2,r3,band_HIGH_PAIRED__Z2  = 0
 c3992: - X__b0,r3,band_UNPAIRED + aux_cap_helper__b0,r3,band_UNPAIRED__condX2
        + aux_cap_helper__b0,r3,band_UNPAIRED__condX1  = 0
 c3993: aux_cap_helper__b1,r13,band_LOW_PAIRED__Z2
        - aux_cap_helper__b1,r13,band_LOW_PAIRED__condX2 <= 0
 c3994: - aux_cap_helper__b4,r4,band_LOW_PAIRED__condX1 <= 0
 c3995: 0.927217345653617 aux_cap_helper__b8,r8,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b8,r8,band_HIGH_PAIRED__Z2
        + 1.5453622427 aux_cap_helper__b8,r8,band_HIGH_PAIRED__condX1
        + 1.3908260184 aux_cap_helper__b8,r8,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b8,r8,band_HIGH_PAIRED__condX2
        - aux_cap_b8,r8,band_HIGH_PAIRED  = 0
 c3996: - 6.37162279776951 sv_function_helpervar_b0,r1__Z2
        + sv_function_helpervar_b0,r1__condX2 <= 0
 c3997: - aux_cap_b7,r5,band_HIGH_PAIRED
        + 0.927217345653617 aux_cap_helper__b7,r5,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b7,r5,band_HIGH_PAIRED__Z2
        + 1.3908260184 aux_cap_helper__b7,r5,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b7,r5,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b7,r5,band_HIGH_PAIRED__condX1  = 0
 c3998: aux_cap_helper__b4,r6,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b4,r6,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b4,r6,band_HIGH_PAIRED__Z3  = 1
 c3999: 0.927217345653617 aux_cap_helper__b1,r0,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b1,r0,band_HIGH_PAIRED__Z2
        + 1.3908260184 aux_cap_helper__b1,r0,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b1,r0,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b1,r0,band_HIGH_PAIRED__condX1
        - aux_cap_b1,r0,band_HIGH_PAIRED  = 0
 c4000: aux_cap_helper__b6,r4,band_UNPAIRED__condX2
        + aux_cap_helper__b6,r4,band_UNPAIRED__condX1 - X__b6,r4,band_UNPAIRED
         = 0
 c4001: 2 aux_cap_helper__b9,r8,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b9,r8,band_HIGH_PAIRED__condX3 <= 0
 c4002: - aux_cap_b0,r7,band_UNPAIRED
        - 0.3214355447 aux_cap_helper__b0,r7,band_UNPAIRED__Z2
        + 1.1250244065 aux_cap_helper__b0,r7,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b0,r7,band_UNPAIRED__condX1  = 0
 c4003: aux_cap_helper__b2,r12,band_LOW_PAIRED__Z2
        - aux_cap_helper__b2,r12,band_LOW_PAIRED__condX2 <= 0
 c4004: aux_cap_helper__b5,r0,band_UNPAIRED__Z1
        + aux_cap_helper__b5,r0,band_UNPAIRED__Z2  = 1
 c4005: aux_cap_helper__b2,r5,band_UNPAIRED__Z2
        + aux_cap_helper__b2,r5,band_UNPAIRED__Z1  = 1
 c4006: - aux_cap_helper__b2,r3,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b2,r3,band_HIGH_PAIRED__Z3 <= 0
 c4007: - sv_function_helpervar_b6,r8__condX1 <= 0
 c4008: - 2 aux_cap_helper__b8,r9,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b8,r9,band_HIGH_PAIRED__condX2 <= 0
 c4009: - 15.9946187432626 sv_function_helpervar_b4,r3__Z3
        + sv_function_helpervar_b4,r3__condX3 <= 0
 c4010: aux_cap_helper__b9,r12,band_LOW_PAIRED__Z2
        + aux_cap_helper__b9,r12,band_LOW_PAIRED__Z1  = 1
 c4011: aux_cap_helper__b1,r13,band_LOW_PAIRED__Z2
        + aux_cap_helper__b1,r13,band_LOW_PAIRED__Z1  = 1
 c4012: 3.735962531 aux_cap_helper__b8,r3,band_LOW_PAIRED__condX1
        + 5.2303475434 aux_cap_helper__b8,r3,band_LOW_PAIRED__condX2
        - 1.4943850124 aux_cap_helper__b8,r3,band_LOW_PAIRED__Z2
        - aux_cap_b8,r3,band_LOW_PAIRED  = 0
 c4013: - X__b7,r2,band_HIGH_PAIRED
        + aux_cap_helper__b7,r2,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b7,r2,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b7,r2,band_HIGH_PAIRED__condX1  = 0
 c4014: - 0.159946187535603 sv_function_helpervar_b0,r6__Z1
        + sv_function_helpervar_b0,r6__condX1 <= 0
 c4015: - 2 aux_cap_helper__b2,r5,band_LOW_PAIRED__Z2
        + aux_cap_helper__b2,r5,band_LOW_PAIRED__condX2 <= 0
 c4016: - sv_function_helpervar_b3,r2__condX1 <= 0
 c4017: sv_function_helpervar_b4,r4__condX1
        - 0.159946187455554 sv_function_helpervar_b4,r4__Z1 <= 0
 c4018: aux_cap_helper__b2,r6,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b2,r6,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b2,r6,band_HIGH_PAIRED__Z1  = 1
 c4019: sv_function_helpervar_b9,r4__condX3
        - 15.9946187432626 sv_function_helpervar_b9,r4__Z3 <= 0
 c4020: - aux_cap_helper__b4,r7,band_HIGH_PAIRED__condX1 <= 0
 c4021: aux_cap_helper__b7,r12,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b7,r12,band_UNPAIRED__Z2 <= 0
 c4022: sv_function_helpervar_b8,r8__condX2
        - 6.66714254024684 sv_function_helpervar_b8,r8__Z2 <= 0
 c4023: 2.1635071398 aux_cap_helper__b6,r4,band_HIGH_PAIRED__condX2
        + 1.3908260184 aux_cap_helper__b6,r4,band_HIGH_PAIRED__condX3
        + 1.5453622427 aux_cap_helper__b6,r4,band_HIGH_PAIRED__condX1
        + 0.927217345653617 aux_cap_helper__b6,r4,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b6,r4,band_HIGH_PAIRED__Z2
        - aux_cap_b6,r4,band_HIGH_PAIRED  = 0
 c4024: aux_cap_helper__b8,r3,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b8,r3,band_HIGH_PAIRED__Z3 <= 0
 c4025: sv_function_helpervar_b4,r10__Z2 + sv_function_helpervar_b4,r10__Z3
        + sv_function_helpervar_b4,r10__Z1  = 1
 c4026: - X__b4,r2,band_HIGH_PAIRED
        + aux_cap_helper__b4,r2,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b4,r2,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b4,r2,band_HIGH_PAIRED__condX1  = 0
 c4027: aux_cap_helper__b5,r12,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b5,r12,band_HIGH_PAIRED__Z2 <= 0
 c4028: - 3 aux_cap_helper__b5,r3,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b5,r3,band_HIGH_PAIRED__condX3 <= 0
 c4029: sv_function_helpervar_b0,r8__Z3 + sv_function_helpervar_b0,r8__Z1
        + sv_function_helpervar_b0,r8__Z2  = 1
 c4030: aux_cap_helper__b0,r5,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b0,r5,band_HIGH_PAIRED__Z3 <= 0
 c4031: - aux_cap_helper__b7,r7,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b7,r7,band_HIGH_PAIRED__Z2 <= 0
 c4032: sv_function_helpervar_b7,r7__condX1
        - 0.159946187465588 sv_function_helpervar_b7,r7__Z1 <= 0
 c4033: aux_cap_helper__b5,r6,band_LOW_PAIRED__condX1
        - aux_cap_helper__b5,r6,band_LOW_PAIRED__Z1 <= 0
 c4034: - sv_function_helpervar_b4,r11__condX1 <= 0
 c4035: - 0.6181448971 aux_cap_helper__b3,r4,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b3,r4,band_HIGH_PAIRED__Z3
        + 1.5453622427 aux_cap_helper__b3,r4,band_HIGH_PAIRED__condX1
        + 2.1635071398 aux_cap_helper__b3,r4,band_HIGH_PAIRED__condX2
        + 1.3908260184 aux_cap_helper__b3,r4,band_HIGH_PAIRED__condX3
        - aux_cap_b3,r4,band_HIGH_PAIRED  = 0
 c4036: aux_cap_helper__b3,r2,band_UNPAIRED__condX1
        - aux_cap_helper__b3,r2,band_UNPAIRED__Z1 <= 0
 c4037: X__b7,r13,band_HIGH_PAIRED + X__b7,r13,band_UNPAIRED
        + X__b7,r13,band_LOW_PAIRED - GlobalBidder_W_ir__b7,r13 >= 0
 c4038: - 2 aux_cap_helper__b8,r6,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b8,r6,band_HIGH_PAIRED__condX2 <= 0
 c4039: sv_function_helpervar_b3,r13__condX3
        - 15.9946187432626 sv_function_helpervar_b3,r13__Z3 <= 0
 c4040: - aux_cap_helper__b7,r7,band_HIGH_PAIRED__condX1 <= 0
 c4041: aux_cap_helper__b6,r2,band_LOW_PAIRED__Z2
        - aux_cap_helper__b6,r2,band_LOW_PAIRED__condX2 <= 0
 c4042: - 3 aux_cap_helper__b8,r4,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b8,r4,band_HIGH_PAIRED__condX3 <= 0
 c4043: aux_cap_helper__b0,r13,band_UNPAIRED__Z1
        + aux_cap_helper__b0,r13,band_UNPAIRED__Z2  = 1
 c4044: - aux_cap_helper__b3,r9,band_UNPAIRED__condX1 <= 0
 c4045: - aux_cap_helper__b0,r11,band_UNPAIRED__condX1 <= 0
 c4046: - 15.9946187432626 sv_function_helpervar_b5,r2__Z3
        + sv_function_helpervar_b5,r2__condX3 <= 0
 c4047: 1.5453622427 aux_cap_helper__b9,r1,band_HIGH_PAIRED__condX1
        + 2.1635071398 aux_cap_helper__b9,r1,band_HIGH_PAIRED__condX2
        + 1.3908260184 aux_cap_helper__b9,r1,band_HIGH_PAIRED__condX3
        - aux_cap_b9,r1,band_HIGH_PAIRED
        + 0.927217345653617 aux_cap_helper__b9,r1,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b9,r1,band_HIGH_PAIRED__Z2  = 0
 c4048: aux_cap_helper__b7,r1,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b7,r1,band_HIGH_PAIRED__condX2 <= 0
 c4049: aux_cap_helper__b7,r1,band_LOW_PAIRED__condX1
        - aux_cap_helper__b7,r1,band_LOW_PAIRED__Z1 <= 0
 c4050: sv_function_helpervar_b2,r2__condX1
        - 0.159946187457741 sv_function_helpervar_b2,r2__Z1 <= 0
 c4051: sv_function_helpervar_b4,r3__Z1 + sv_function_helpervar_b4,r3__Z2
        + sv_function_helpervar_b4,r3__Z3  = 1
 c4052: - sv_function_helpervar_b4,r5__condX1 <= 0
 c4053: - aux_cap_helper__b5,r1,band_HIGH_PAIRED__condX1 <= 0
 c4054: - sv_function_helpervar_b6,r11__condX2
        + 0.159946187466827 sv_function_helpervar_b6,r11__Z2 <= 0
 c4055: - X__b3,r7,band_LOW_PAIRED
        + aux_cap_helper__b3,r7,band_LOW_PAIRED__condX2
        + aux_cap_helper__b3,r7,band_LOW_PAIRED__condX1  = 0
 c4056: sv_function_helpervar_b3,r7__Z2 + sv_function_helpervar_b3,r7__Z1
        + sv_function_helpervar_b3,r7__Z3  = 1
 c4057: aux_cap_helper__b1,r2,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b1,r2,band_LOW_PAIRED__Z2 <= 0
 c4058: GlobalBidder_W_i_7 - 14 GlobalBidder_hat{W}__i7,k1 >= -1
 c4059: aux_cap_helper__b0,r5,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b0,r5,band_HIGH_PAIRED__Z1 <= 0
 c4060: aux_cap_helper__b3,r12,band_LOW_PAIRED__condX1
        - aux_cap_helper__b3,r12,band_LOW_PAIRED__Z1 <= 0
 c4061: - X__b0,r1,band_UNPAIRED + aux_cap_helper__b0,r1,band_UNPAIRED__condX1
        + aux_cap_helper__b0,r1,band_UNPAIRED__condX2  = 0
 c4062: - aux_cap_helper__b4,r13,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b4,r13,band_HIGH_PAIRED__condX1 <= 0
 c4063: 6.67209283899866 sv_function_helpervar_b7,r9__Z3
        - sv_function_helpervar_b7,r9__condX3 <= 0
 c4064: - 0.3214355447 aux_cap_helper__b0,r13,band_UNPAIRED__Z2
        - aux_cap_b0,r13,band_UNPAIRED
        + 1.1250244065 aux_cap_helper__b0,r13,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b0,r13,band_UNPAIRED__condX1  = 0
 c4065: - X__b6,r5,band_UNPAIRED + aux_cap_helper__b6,r5,band_UNPAIRED__condX1
        + aux_cap_helper__b6,r5,band_UNPAIRED__condX2  = 0
 c4066: - aux_Omega_b2,r4 + 1910.72907962888 aux_sv_b2,r4  = 0
 c4067: - aux_cap_b4,r0,band_HIGH_PAIRED
        + 1.3908260184 aux_cap_helper__b4,r0,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b4,r0,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b4,r0,band_HIGH_PAIRED__condX1
        - 0.6181448971 aux_cap_helper__b4,r0,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b4,r0,band_HIGH_PAIRED__Z3  = 0
 c4068: - aux_sv_b2,r13 + 24.6330231183191 sv_function_helpervar_b2,r13__Z2
        + 51.5816280939243 sv_function_helpervar_b2,r13__Z3
        + 160.836769383 sv_function_helpervar_b2,r13__condX1
        + 6.8285775799 sv_function_helpervar_b2,r13__condX2
        + 2.731980992 sv_function_helpervar_b2,r13__condX3  = 0
 c4069: - sv_function_helpervar_b5,r3__condX1 <= 0
 c4070: - aux_cap_b4,r12,band_LOW_PAIRED
        - 1.4943850124 aux_cap_helper__b4,r12,band_LOW_PAIRED__Z2
        + 5.2303475434 aux_cap_helper__b4,r12,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b4,r12,band_LOW_PAIRED__condX1  = 0
 c4071: - sv_function_helpervar_b2,r0__condX3
        + 6.18862644261808 sv_function_helpervar_b2,r0__Z3 <= 0
 c4072: aux_cap_helper__b5,r5,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b5,r5,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b5,r5,band_HIGH_PAIRED__Z1  = 1
 c4073: aux_cap_helper__b9,r0,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b9,r0,band_UNPAIRED__Z2 <= 0
 c4074: aux_cap_helper__b1,r11,band_UNPAIRED__Z2
        + aux_cap_helper__b1,r11,band_UNPAIRED__Z1  = 1
 c4075: - aux_cap_helper__b3,r7,band_LOW_PAIRED__condX1 <= 0
 c4076: sv_function_helpervar_b9,r13__condX1
        - 0.159946187446348 sv_function_helpervar_b9,r13__Z1 <= 0
 c4077: - aux_Omega_b0,r13 + 30.5456130212447 aux_sv_b0,r13  = 0
 c4078: - aux_cap_helper__b4,r13,band_LOW_PAIRED__condX1 <= 0
 c4079: - X__b7,r8,band_UNPAIRED + aux_cap_helper__b7,r8,band_UNPAIRED__condX1
        + aux_cap_helper__b7,r8,band_UNPAIRED__condX2  = 0
 c4080: aux_cap_helper__b3,r0,band_LOW_PAIRED__condX1
        + aux_cap_helper__b3,r0,band_LOW_PAIRED__condX2
        - X__b3,r0,band_LOW_PAIRED  = 0
 c4081: aux_cap_helper__b6,r0,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b6,r0,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b6,r0,band_HIGH_PAIRED__condX2
        - X__b6,r0,band_HIGH_PAIRED  = 0
 c4082: - X__b0,r3,band_HIGH_PAIRED
        + aux_cap_helper__b0,r3,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b0,r3,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b0,r3,band_HIGH_PAIRED__condX1  = 0
 c4083: - 1.4943850124 aux_cap_helper__b6,r11,band_LOW_PAIRED__Z2
        + 5.2303475434 aux_cap_helper__b6,r11,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b6,r11,band_LOW_PAIRED__condX1
        - aux_cap_b6,r11,band_LOW_PAIRED  = 0
 c4084: - 0.6181448971 aux_cap_helper__b3,r0,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b3,r0,band_HIGH_PAIRED__Z3
        - aux_cap_b3,r0,band_HIGH_PAIRED
        + 1.3908260184 aux_cap_helper__b3,r0,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b3,r0,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b3,r0,band_HIGH_PAIRED__condX1  = 0
 c4085: sv_function_helpervar_b0,r0__Z3 + sv_function_helpervar_b0,r0__Z2
        + sv_function_helpervar_b0,r0__Z1  = 1
 c4086: 2 aux_cap_helper__b1,r12,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b1,r12,band_HIGH_PAIRED__condX3 <= 0
 c4087: - X__b3,r2,band_UNPAIRED + aux_cap_helper__b3,r2,band_UNPAIRED__condX2
        + aux_cap_helper__b3,r2,band_UNPAIRED__condX1  = 0
 c4088: 2 aux_cap_helper__b4,r5,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b4,r5,band_HIGH_PAIRED__condX3 <= 0
 c4089: - aux_cap_helper__b6,r2,band_LOW_PAIRED__condX1 <= 0
 c4090: - aux_cap_helper__b7,r1,band_HIGH_PAIRED__condX1 <= 0
 c4091: - X__b6,r11,band_LOW_PAIRED
        + aux_cap_helper__b6,r11,band_LOW_PAIRED__condX2
        + aux_cap_helper__b6,r11,band_LOW_PAIRED__condX1  = 0
 c4092: aux_cap_helper__b8,r12,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b8,r12,band_HIGH_PAIRED__condX2 <= 0
 c4093: aux_cap_helper__b0,r3,band_UNPAIRED__Z2
        + aux_cap_helper__b0,r3,band_UNPAIRED__Z1  = 1
 c4094: - aux_Omega_b6,r12 + 862.552757354164 aux_sv_b6,r12  = 0
 c4095: aux_cap_helper__b4,r5,band_UNPAIRED__Z2
        - aux_cap_helper__b4,r5,band_UNPAIRED__condX2 <= 0
 c4096: - aux_cap_helper__b5,r8,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b5,r8,band_HIGH_PAIRED__condX1 <= 0
 c4097: - aux_cap_b5,r1,band_HIGH_PAIRED
        - 0.6181448971 aux_cap_helper__b5,r1,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b5,r1,band_HIGH_PAIRED__Z3
        + 1.5453622427 aux_cap_helper__b5,r1,band_HIGH_PAIRED__condX1
        + 2.1635071398 aux_cap_helper__b5,r1,band_HIGH_PAIRED__condX2
        + 1.3908260184 aux_cap_helper__b5,r1,band_HIGH_PAIRED__condX3  = 0
 c4098: aux_cap_helper__b7,r8,band_LOW_PAIRED__Z1
        + aux_cap_helper__b7,r8,band_LOW_PAIRED__Z2  = 1
 c4099: - 0.3214355447 aux_cap_helper__b8,r0,band_UNPAIRED__Z2
        + 0.8035888618 aux_cap_helper__b8,r0,band_UNPAIRED__condX1
        + 1.1250244065 aux_cap_helper__b8,r0,band_UNPAIRED__condX2
        - aux_cap_b8,r0,band_UNPAIRED  = 0
 c4100: - sv_function_helpervar_b4,r10__condX1 <= 0
 c4101: aux_cap_helper__b6,r4,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b6,r4,band_HIGH_PAIRED__Z2 <= 0
 c4102: X__b7,r5,band_UNPAIRED + X__b7,r5,band_HIGH_PAIRED
        - GlobalBidder_W_ir__b7,r5 + X__b7,r5,band_LOW_PAIRED >= 0
 c4103: - aux_cap_helper__b8,r12,band_HIGH_PAIRED__condX1 <= 0
 c4104: aux_cap_helper__b1,r7,band_UNPAIRED__Z1
        + aux_cap_helper__b1,r7,band_UNPAIRED__Z2  = 1
 c4105: sv_function_helpervar_b4,r4__condX3
        - 15.9946187432626 sv_function_helpervar_b4,r4__Z3 <= 0
 c4106: - aux_cap_helper__b4,r5,band_UNPAIRED__condX1 <= 0
 c4107: aux_cap_helper__b5,r4,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b5,r4,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b5,r4,band_HIGH_PAIRED__condX3
        - X__b5,r4,band_HIGH_PAIRED  = 0
 c4108: - X__b1,r4,band_UNPAIRED + aux_cap_helper__b1,r4,band_UNPAIRED__condX1
        + aux_cap_helper__b1,r4,band_UNPAIRED__condX2  = 0
 c4109: aux_cap_helper__b3,r1,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b3,r1,band_HIGH_PAIRED__Z3 <= 0
 c4110: - aux_Omega_b6,r13 + 14.3846663535252 aux_sv_b6,r13  = 0
 c4111: - aux_cap_helper__b3,r9,band_UNPAIRED__condX2
        + aux_cap_helper__b3,r9,band_UNPAIRED__Z2 <= 0
 c4112: - 6.02685562864558 sv_function_helpervar_b5,r6__Z2
        + sv_function_helpervar_b5,r6__condX2 <= 0
 c4113: aux_cap_helper__b1,r7,band_LOW_PAIRED__condX1
        - aux_cap_helper__b1,r7,band_LOW_PAIRED__Z1 <= 0
 c4114: aux_cap_helper__b1,r0,band_LOW_PAIRED__Z2
        - aux_cap_helper__b1,r0,band_LOW_PAIRED__condX2 <= 0
 c4115: - 0.159946187464243 sv_function_helpervar_b5,r11__Z1
        + sv_function_helpervar_b5,r11__condX1 <= 0
 c4116: aux_cap_helper__b8,r3,band_LOW_PAIRED__Z2
        + aux_cap_helper__b8,r3,band_LOW_PAIRED__Z1  = 1
 c4117: sv_function_helpervar_b8,r10__condX2
        - 7.6310207645995 sv_function_helpervar_b8,r10__Z2 <= 0
 c4118: sv_function_helpervar_b7,r6__condX2
        - 7.56076262079736 sv_function_helpervar_b7,r6__Z2 <= 0
 c4119: aux_cap_helper__b3,r3,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b3,r3,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b3,r3,band_HIGH_PAIRED__Z1  = 1
 c4120: aux_cap_helper__b8,r10,band_LOW_PAIRED__Z1
        + aux_cap_helper__b8,r10,band_LOW_PAIRED__Z2  = 1
 c4121: 0.159946187447607 sv_function_helpervar_b0,r4__Z2
        - sv_function_helpervar_b0,r4__condX2 <= 0
 c4122: sv_function_helpervar_b1,r11__condX3
        - 15.9946187432626 sv_function_helpervar_b1,r11__Z3 <= 0
 c4123: aux_cap_helper__b5,r5,band_LOW_PAIRED__Z2
        - aux_cap_helper__b5,r5,band_LOW_PAIRED__condX2 <= 0
 c4124: aux_cap_helper__b1,r7,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b1,r7,band_UNPAIRED__Z2 <= 0
 c4125: aux_cap_helper__b4,r5,band_LOW_PAIRED__Z2
        + aux_cap_helper__b4,r5,band_LOW_PAIRED__Z1  = 1
 c4126: - aux_cap_helper__b8,r3,band_UNPAIRED__condX2
        + aux_cap_helper__b8,r3,band_UNPAIRED__Z2 <= 0
 c4127: - sv_function_helpervar_b1,r12__condX2
        + 0.15994618744495 sv_function_helpervar_b1,r12__Z2 <= 0
 c4128: - 0.159946187433943 sv_function_helpervar_b1,r0__Z1
        + sv_function_helpervar_b1,r0__condX1 <= 0
 c4129: 35.0909663637 sv_function_helpervar_b7,r6__condX3
        + 68.1296176206 sv_function_helpervar_b7,r6__condX2
        + 1850.3233256754 sv_function_helpervar_b7,r6__condX1
        + 285.055088918132 sv_function_helpervar_b7,r6__Z2
        + 534.852488382849 sv_function_helpervar_b7,r6__Z3 - aux_sv_b7,r6  = 0
 c4130: aux_cap_helper__b9,r10,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b9,r10,band_UNPAIRED__Z2 <= 0
 c4131: aux_cap_b2,r9,band_UNPAIRED - aux_c_b2,r9
        + aux_cap_b2,r9,band_LOW_PAIRED + aux_cap_b2,r9,band_HIGH_PAIRED  = 0
 c4132: 2.1635071398 aux_cap_helper__b3,r1,band_HIGH_PAIRED__condX2
        + 1.3908260184 aux_cap_helper__b3,r1,band_HIGH_PAIRED__condX3
        + 1.5453622427 aux_cap_helper__b3,r1,band_HIGH_PAIRED__condX1
        + 0.927217345653617 aux_cap_helper__b3,r1,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b3,r1,band_HIGH_PAIRED__Z2
        - aux_cap_b3,r1,band_HIGH_PAIRED  = 0
 c4133: aux_cap_helper__b2,r4,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b2,r4,band_HIGH_PAIRED__Z3 <= 0
 c4134: - aux_cap_helper__b8,r3,band_UNPAIRED__condX1 <= 0
 c4135: aux_cap_helper__b5,r5,band_LOW_PAIRED__condX2
        + aux_cap_helper__b5,r5,band_LOW_PAIRED__condX1
        - X__b5,r5,band_LOW_PAIRED  = 0
 c4136: X__b9,r8,band_HIGH_PAIRED - GlobalBidder_W_ir__b9,r8
        + X__b9,r8,band_UNPAIRED + X__b9,r8,band_LOW_PAIRED >= 0
 c4137: aux_cap_helper__b7,r7,band_UNPAIRED__condX1
        - aux_cap_helper__b7,r7,band_UNPAIRED__Z1 <= 0
 c4138: X__b9,r2,band_UNPAIRED + X__b9,r2,band_HIGH_PAIRED
        - GlobalBidder_W_ir__b9,r2 + X__b9,r2,band_LOW_PAIRED >= 0
 c4139: - sv_function_helpervar_b2,r10__condX1 <= 0
 c4140: - 0.159946187437218 sv_function_helpervar_b1,r8__Z1
        + sv_function_helpervar_b1,r8__condX1 <= 0
 c4141: aux_cap_helper__b1,r8,band_UNPAIRED__condX1
        - aux_cap_helper__b1,r8,band_UNPAIRED__Z1 <= 0
 c4142: - 5.81605834545243 sv_function_helpervar_b6,r8__Z2
        + sv_function_helpervar_b6,r8__condX2 <= 0
 c4143: - aux_cap_helper__b4,r1,band_LOW_PAIRED__condX1 <= 0
 c4144: - sv_function_helpervar_b6,r7__condX2
        + 0.159946187457337 sv_function_helpervar_b6,r7__Z2 <= 0
 c4145: - aux_sv_b2,r0 + 160.8367694204 sv_function_helpervar_b2,r0__condX1
        + 7.2699437494 sv_function_helpervar_b2,r0__condX2
        + 2.6234191584 sv_function_helpervar_b2,r0__condX3
        + 24.5624282824984 sv_function_helpervar_b2,r0__Z2
        + 53.3180332326278 sv_function_helpervar_b2,r0__Z3  = 0
 c4146: - aux_c_b0,r12 + aux_cap_b0,r12,band_LOW_PAIRED
        + aux_cap_b0,r12,band_UNPAIRED + aux_cap_b0,r12,band_HIGH_PAIRED  = 0
 c4147: aux_cap_helper__b4,r6,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b4,r6,band_UNPAIRED__Z2 <= 0
 c4148: - 2 aux_cap_helper__b6,r3,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b6,r3,band_HIGH_PAIRED__condX2 <= 0
 c4149: - 2 aux_cap_helper__b2,r3,band_UNPAIRED__Z2
        + aux_cap_helper__b2,r3,band_UNPAIRED__condX2 <= 0
 c4150: - aux_cap_helper__b5,r7,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b5,r7,band_HIGH_PAIRED__condX1 <= 0
 c4151: 2 aux_cap_helper__b8,r13,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b8,r13,band_HIGH_PAIRED__condX3 <= 0
 c4152: sv_function_helpervar_b0,r10__Z2 + sv_function_helpervar_b0,r10__Z3
        + sv_function_helpervar_b0,r10__Z1  = 1
 c4153: sv_function_helpervar_b7,r2__condX3
        - 15.9946187432626 sv_function_helpervar_b7,r2__Z3 <= 0
 c4154: - aux_sv_b1,r2 + 125.5635947181 sv_function_helpervar_b1,r2__condX1
        + 1.964864877 sv_function_helpervar_b1,r2__condX3
        + 6.0954484623 sv_function_helpervar_b1,r2__condX2
        + 42.9557659993454 sv_function_helpervar_b1,r2__Z3
        + 19.1084745164437 sv_function_helpervar_b1,r2__Z2  = 0
 c4155: - aux_cap_helper__b6,r10,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b6,r10,band_HIGH_PAIRED__Z2 <= 0
 c4156: - sv_function_helpervar_b0,r3__condX1 <= 0
 c4157: - X__b4,r12,band_UNPAIRED
        + aux_cap_helper__b4,r12,band_UNPAIRED__condX1
        + aux_cap_helper__b4,r12,band_UNPAIRED__condX2  = 0
 c4158: - 3 aux_cap_helper__b5,r11,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b5,r11,band_HIGH_PAIRED__condX3 <= 0
 c4159: - aux_cap_b5,r11,band_UNPAIRED
        + 0.8035888618 aux_cap_helper__b5,r11,band_UNPAIRED__condX1
        + 1.1250244065 aux_cap_helper__b5,r11,band_UNPAIRED__condX2
        - 0.3214355447 aux_cap_helper__b5,r11,band_UNPAIRED__Z2  = 0
 c4160: 0.927217345653617 aux_cap_helper__b2,r0,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b2,r0,band_HIGH_PAIRED__Z2
        + 1.5453622427 aux_cap_helper__b2,r0,band_HIGH_PAIRED__condX1
        + 1.3908260184 aux_cap_helper__b2,r0,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b2,r0,band_HIGH_PAIRED__condX2
        - aux_cap_b2,r0,band_HIGH_PAIRED  = 0
 c4161: - 0.3214355447 aux_cap_helper__b6,r3,band_UNPAIRED__Z2
        + 1.1250244065 aux_cap_helper__b6,r3,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b6,r3,band_UNPAIRED__condX1
        - aux_cap_b6,r3,band_UNPAIRED  = 0
 c4162: - aux_cap_helper__b8,r13,band_LOW_PAIRED__Z1
        + aux_cap_helper__b8,r13,band_LOW_PAIRED__condX1 <= 0
 c4163: - aux_cap_helper__b1,r0,band_LOW_PAIRED__condX1 <= 0
 c4164: aux_cap_helper__b5,r3,band_LOW_PAIRED__condX1
        - aux_cap_helper__b5,r3,band_LOW_PAIRED__Z1 <= 0
 c4165: 200.057110149321 aux_sv_b0,r8 - aux_Omega_b0,r8  = 0
 c4166: - sv_function_helpervar_b3,r7__condX3
        + 6.24248591508399 sv_function_helpervar_b3,r7__Z3 <= 0
 c4167: - sv_function_helpervar_b1,r0__condX1 <= 0
 c4168: - aux_cap_helper__b6,r10,band_HIGH_PAIRED__condX1 <= 0
 c4169: aux_cap_helper__b9,r11,band_UNPAIRED__Z2
        - aux_cap_helper__b9,r11,band_UNPAIRED__condX2 <= 0
 c4170: aux_cap_helper__b5,r12,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b5,r12,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b5,r12,band_HIGH_PAIRED__Z1  = 1
 c4171: - aux_cap_helper__b5,r5,band_LOW_PAIRED__condX1 <= 0
 c4172: sv_function_helpervar_b9,r12__Z3 + sv_function_helpervar_b9,r12__Z1
        + sv_function_helpervar_b9,r12__Z2  = 1
 c4173: - aux_cap_b0,r7,band_HIGH_PAIRED
        + 1.3908260184 aux_cap_helper__b0,r7,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b0,r7,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b0,r7,band_HIGH_PAIRED__condX1
        + 0.927217345653617 aux_cap_helper__b0,r7,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b0,r7,band_HIGH_PAIRED__Z2  = 0
 c4174: - aux_cap_helper__b4,r1,band_LOW_PAIRED__condX2
        + aux_cap_helper__b4,r1,band_LOW_PAIRED__Z2 <= 0
 c4175: - aux_cap_helper__b2,r13,band_LOW_PAIRED__Z1
        + aux_cap_helper__b2,r13,band_LOW_PAIRED__condX1 <= 0
 c4176: X__b8,r4,band_HIGH_PAIRED + X__b8,r4,band_UNPAIRED
        + X__b8,r4,band_LOW_PAIRED - GlobalBidder_W_ir__b8,r4 >= 0
 c4177: - aux_cap_helper__b9,r11,band_UNPAIRED__condX1 <= 0
 c4178: aux_cap_helper__b5,r2,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b5,r2,band_UNPAIRED__Z2 <= 0
 c4179: aux_cap_helper__b3,r1,band_LOW_PAIRED__Z1
        + aux_cap_helper__b3,r1,band_LOW_PAIRED__Z2  = 1
 c4180: 3.735962531 aux_cap_helper__b5,r0,band_LOW_PAIRED__condX1
        + 5.2303475434 aux_cap_helper__b5,r0,band_LOW_PAIRED__condX2
        - aux_cap_b5,r0,band_LOW_PAIRED
        - 1.4943850124 aux_cap_helper__b5,r0,band_LOW_PAIRED__Z2  = 0
 c4181: - aux_sv_b2,r4 + 2.7320008414 sv_function_helpervar_b2,r4__condX3
        + 160.8367694156 sv_function_helpervar_b2,r4__condX1
        + 6.8285047934 sv_function_helpervar_b2,r4__condX2
        + 24.6330347604573 sv_function_helpervar_b2,r4__Z2
        + 51.5813106105989 sv_function_helpervar_b2,r4__Z3  = 0
 c4182: aux_cap_helper__b5,r1,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b5,r1,band_HIGH_PAIRED__condX2 <= 0
 c4183: aux_cap_helper__b6,r3,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b6,r3,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b6,r3,band_HIGH_PAIRED__condX1
        - X__b6,r3,band_HIGH_PAIRED  = 0
 c4184: sv_function_helpervar_b4,r4__condX2
        - 6.18586368768639 sv_function_helpervar_b4,r4__Z2 <= 0
 c4185: - sv_function_helpervar_b8,r8__condX1 <= 0
 c4186: aux_cap_helper__b1,r7,band_UNPAIRED__condX1
        + aux_cap_helper__b1,r7,band_UNPAIRED__condX2 - X__b1,r7,band_UNPAIRED
         = 0
 c4187: - 5.47156313732081 sv_function_helpervar_b6,r4__Z2
        + sv_function_helpervar_b6,r4__condX2 <= 0
 c4188: - aux_cap_helper__b4,r10,band_UNPAIRED__Z1
        + aux_cap_helper__b4,r10,band_UNPAIRED__condX1 <= 0
 c4189: - X__b7,r5,band_HIGH_PAIRED
        + aux_cap_helper__b7,r5,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b7,r5,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b7,r5,band_HIGH_PAIRED__condX1  = 0
 c4190: sv_function_helpervar_b7,r1__Z1 + sv_function_helpervar_b7,r1__Z2
        + sv_function_helpervar_b7,r1__Z3  = 1
 c4191: - aux_c_b3,r0 + aux_cap_b3,r0,band_HIGH_PAIRED
        + aux_cap_b3,r0,band_UNPAIRED + aux_cap_b3,r0,band_LOW_PAIRED  = 0
 c4192: - 2 aux_cap_helper__b5,r13,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b5,r13,band_HIGH_PAIRED__condX2 <= 0
 c4193: 5.94106823473582 sv_function_helpervar_b0,r6__Z3
        - sv_function_helpervar_b0,r6__condX3 <= 0
 c4194: aux_cap_helper__b8,r6,band_LOW_PAIRED__condX2
        + aux_cap_helper__b8,r6,band_LOW_PAIRED__condX1
        - X__b8,r6,band_LOW_PAIRED  = 0
 c4195: - 15.9946187432626 sv_function_helpervar_b6,r10__Z3
        + sv_function_helpervar_b6,r10__condX3 <= 0
 c4196: - 0.3214355447 aux_cap_helper__b7,r11,band_UNPAIRED__Z2
        - aux_cap_b7,r11,band_UNPAIRED
        + 0.8035888618 aux_cap_helper__b7,r11,band_UNPAIRED__condX1
        + 1.1250244065 aux_cap_helper__b7,r11,band_UNPAIRED__condX2  = 0
 c4197: - aux_cap_helper__b1,r0,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b1,r0,band_HIGH_PAIRED__condX1 <= 0
 c4198: - aux_cap_helper__b7,r13,band_UNPAIRED__condX1 <= 0
 c4199: - aux_cap_helper__b5,r5,band_UNPAIRED__condX1 <= 0
 c4200: X__b0,r2,band_HIGH_PAIRED + X__b7,r2,band_HIGH_PAIRED
        + X__b4,r2,band_HIGH_PAIRED + X__b1,r2,band_HIGH_PAIRED
        + X__b3,r2,band_HIGH_PAIRED + X__b9,r2,band_HIGH_PAIRED
        + X__b2,r2,band_HIGH_PAIRED + X__b5,r2,band_HIGH_PAIRED
        + X__b6,r2,band_HIGH_PAIRED + X__b8,r2,band_HIGH_PAIRED <= 3
 c4201: aux_cap_b2,r2,band_HIGH_PAIRED - aux_c_b2,r2
        + aux_cap_b2,r2,band_LOW_PAIRED + aux_cap_b2,r2,band_UNPAIRED  = 0
 c4202: - aux_c_b3,r13 + aux_cap_b3,r13,band_LOW_PAIRED
        + aux_cap_b3,r13,band_HIGH_PAIRED + aux_cap_b3,r13,band_UNPAIRED  = 0
 c4203: - 15.9946187432626 sv_function_helpervar_b7,r4__Z3
        + sv_function_helpervar_b7,r4__condX3 <= 0
 c4204: sv_function_helpervar_b9,r7__condX1
        - 0.159946187435082 sv_function_helpervar_b9,r7__Z1 <= 0
 c4205: aux_cap_helper__b0,r4,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b0,r4,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b0,r4,band_HIGH_PAIRED__Z1  = 1
 c4206: aux_cap_helper__b6,r9,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b6,r9,band_HIGH_PAIRED__condX2 <= 0
 c4207: - 2 aux_cap_helper__b3,r3,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b3,r3,band_HIGH_PAIRED__condX2 <= 0
 c4208: - sv_function_helpervar_b2,r3__condX1 <= 0
 c4209: 1719.0945878091 sv_function_helpervar_b9,r4__condX1
        + 29.0365528325 sv_function_helpervar_b9,r4__condX3
        + 73.5969596385 sv_function_helpervar_b9,r4__condX2
        + 263.191072129382 sv_function_helpervar_b9,r4__Z2
        + 553.951501283864 sv_function_helpervar_b9,r4__Z3 - aux_sv_b9,r4  = 0
 c4210: - aux_cap_helper__b7,r6,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b7,r6,band_HIGH_PAIRED__Z3 <= 0
 c4211: - 15.9946187432626 sv_function_helpervar_b1,r13__Z3
        + sv_function_helpervar_b1,r13__condX3 <= 0
 c4212: aux_cap_helper__b6,r11,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b6,r11,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b6,r11,band_HIGH_PAIRED__Z1  = 1
 c4213: aux_cap_helper__b7,r4,band_LOW_PAIRED__condX1
        - aux_cap_helper__b7,r4,band_LOW_PAIRED__Z1 <= 0
 c4214: aux_cap_helper__b7,r2,band_UNPAIRED__Z1
        + aux_cap_helper__b7,r2,band_UNPAIRED__Z2  = 1
 c4215: aux_cap_helper__b5,r0,band_LOW_PAIRED__condX1
        - aux_cap_helper__b5,r0,band_LOW_PAIRED__Z1 <= 0
 c4216: - 2 aux_cap_helper__b8,r7,band_UNPAIRED__Z2
        + aux_cap_helper__b8,r7,band_UNPAIRED__condX2 <= 0
 c4217: sv_function_helpervar_b5,r10__condX1
        + sv_function_helpervar_b5,r10__condX2
        + sv_function_helpervar_b5,r10__condX3 - aux_c_b5,r10  = 0
 c4218: sv_function_helpervar_b1,r6__condX1
        + sv_function_helpervar_b1,r6__condX3
        + sv_function_helpervar_b1,r6__condX2 - aux_c_b1,r6  = 0
 c4219: - aux_cap_helper__b7,r4,band_UNPAIRED__condX2
        + aux_cap_helper__b7,r4,band_UNPAIRED__Z2 <= 0
 c4220: 53.9756608176 sv_function_helpervar_b8,r6__condX2
        + 1343.0085755247 sv_function_helpervar_b8,r6__condX1
        + 23.7243443463 sv_function_helpervar_b8,r6__condX3 - aux_sv_b8,r6
        + 416.127421516165 sv_function_helpervar_b8,r6__Z3
        + 206.1759001903 sv_function_helpervar_b8,r6__Z2  = 0
 c4221: - aux_cap_helper__b6,r9,band_HIGH_PAIRED__condX1 <= 0
 c4222: - sv_function_helpervar_b6,r11__condX1 <= 0
 c4223: sv_function_helpervar_b8,r7__condX3
        - 15.9946187432626 sv_function_helpervar_b8,r7__Z3 <= 0
 c4224: aux_cap_helper__b7,r4,band_UNPAIRED__Z2
        + aux_cap_helper__b7,r4,band_UNPAIRED__Z1  = 1
 c4225: - aux_cap_helper__b6,r8,band_UNPAIRED__condX1 <= 0
 c4226: 0.0102040816326531 X__b7,r13,band_HIGH_PAIRED
        + 0.0102040816326531 X__b7,r13,band_UNPAIRED
        + 0.0102040816326531 X__b7,r13,band_LOW_PAIRED
        - GlobalBidder_W_ir__b7,r13 <= 0
 c4227: aux_cap_helper__b1,r5,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b1,r5,band_LOW_PAIRED__Z2 <= 0
 c4228: - aux_cap_helper__b4,r6,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b4,r6,band_HIGH_PAIRED__Z3 <= 0
 c4229: - 0.6181448971 aux_cap_helper__b8,r4,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b8,r4,band_HIGH_PAIRED__Z3
        - aux_cap_b8,r4,band_HIGH_PAIRED
        + 1.5453622427 aux_cap_helper__b8,r4,band_HIGH_PAIRED__condX1
        + 2.1635071398 aux_cap_helper__b8,r4,band_HIGH_PAIRED__condX2
        + 1.3908260184 aux_cap_helper__b8,r4,band_HIGH_PAIRED__condX3  = 0
 c4230: - aux_cap_helper__b3,r9,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b3,r9,band_HIGH_PAIRED__condX1 <= 0
 c4231: - 1.4943850124 aux_cap_helper__b6,r3,band_LOW_PAIRED__Z2
        - aux_cap_b6,r3,band_LOW_PAIRED
        + 3.735962531 aux_cap_helper__b6,r3,band_LOW_PAIRED__condX1
        + 5.2303475434 aux_cap_helper__b6,r3,band_LOW_PAIRED__condX2  = 0
 c4232: 2.1442869292 sv_function_helpervar_b1,r9__condX3
        + 125.5635947313 sv_function_helpervar_b1,r9__condX1
        + 5.2895335301 sv_function_helpervar_b1,r9__condX2 - aux_sv_b1,r9
        + 40.0859786800774 sv_function_helpervar_b1,r9__Z3
        + 19.2373775373649 sv_function_helpervar_b1,r9__Z2  = 0
 c4233: aux_cap_helper__b5,r1,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b5,r1,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b5,r1,band_HIGH_PAIRED__condX3
        - X__b5,r1,band_HIGH_PAIRED  = 0
 c4234: 359.834085583931 sv_function_helpervar_b6,r10__Z3
        + 161.375896006363 sv_function_helpervar_b6,r10__Z2 - aux_sv_b6,r10
        + 16.742502001 sv_function_helpervar_b6,r10__condX3
        + 50.533182628 sv_function_helpervar_b6,r10__condX2
        + 1059.4718668243 sv_function_helpervar_b6,r10__condX1  = 0
 c4235: aux_cap_helper__b9,r3,band_UNPAIRED__Z2
        - aux_cap_helper__b9,r3,band_UNPAIRED__condX2 <= 0
 c4236: - aux_cap_helper__b7,r4,band_UNPAIRED__condX1 <= 0
 c4237: - aux_c_b3,r5 + sv_function_helpervar_b3,r5__condX1
        + sv_function_helpervar_b3,r5__condX2
        + sv_function_helpervar_b3,r5__condX3  = 0
 c4238: - aux_cap_helper__b8,r10,band_LOW_PAIRED__Z1
        + aux_cap_helper__b8,r10,band_LOW_PAIRED__condX1 <= 0
 c4239: - aux_cap_helper__b2,r7,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b2,r7,band_HIGH_PAIRED__Z3 <= 0
 c4240: X__b2,r2,band_LOW_PAIRED + X__b6,r2,band_LOW_PAIRED
        + X__b0,r2,band_LOW_PAIRED + X__b4,r2,band_LOW_PAIRED
        + X__b3,r2,band_LOW_PAIRED + X__b1,r2,band_LOW_PAIRED
        + X__b5,r2,band_LOW_PAIRED + X__b7,r2,band_LOW_PAIRED
        + X__b8,r2,band_LOW_PAIRED + X__b9,r2,band_LOW_PAIRED <= 2
 c4241: 125.5635947333 sv_function_helpervar_b1,r4__condX1
        + 5.605599471 sv_function_helpervar_b1,r4__condX2
        + 2.0639145295 sv_function_helpervar_b1,r4__condX3
        + 41.3715045706506 sv_function_helpervar_b1,r4__Z3
        + 19.1868239951393 sv_function_helpervar_b1,r4__Z2 - aux_sv_b1,r4  = 0
 c4242: aux_cap_helper__b0,r4,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b0,r4,band_HIGH_PAIRED__Z2 <= 0
 c4243: - aux_cap_helper__b9,r9,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b9,r9,band_HIGH_PAIRED__Z2 <= 0
 c4244: - 2 aux_cap_helper__b8,r0,band_UNPAIRED__Z2
        + aux_cap_helper__b8,r0,band_UNPAIRED__condX2 <= 0
 c4245: - X__b9,r4,band_UNPAIRED + aux_cap_helper__b9,r4,band_UNPAIRED__condX1
        + aux_cap_helper__b9,r4,band_UNPAIRED__condX2  = 0
 c4246: - aux_cap_helper__b6,r5,band_LOW_PAIRED__condX1 <= 0
 c4247: - aux_cap_helper__b9,r3,band_UNPAIRED__condX1 <= 0
 c4248: - aux_Omega_b6,r1 + 216.387523173414 aux_sv_b6,r1  = 0
 c4249: aux_cap_helper__b0,r1,band_LOW_PAIRED__condX2
        + aux_cap_helper__b0,r1,band_LOW_PAIRED__condX1
        - X__b0,r1,band_LOW_PAIRED  = 0
 c4250: 24.6421340581381 sv_function_helpervar_b2,r6__Z2
        + 51.3296357341762 sv_function_helpervar_b2,r6__Z3 - aux_sv_b2,r6
        + 6.7716150493 sv_function_helpervar_b2,r6__condX2
        + 160.8367694155 sv_function_helpervar_b2,r6__condX1
        + 2.7477358133 sv_function_helpervar_b2,r6__condX3  = 0
 c4251: aux_cap_helper__b7,r13,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b7,r13,band_HIGH_PAIRED__Z3 <= 0
 c4252: - sv_function_helpervar_b4,r13__condX3
        + 6.03557477569176 sv_function_helpervar_b4,r13__Z3 <= 0
 c4253: - aux_cap_b1,r4,band_LOW_PAIRED
        + 5.2303475434 aux_cap_helper__b1,r4,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b1,r4,band_LOW_PAIRED__condX1
        - 1.4943850124 aux_cap_helper__b1,r4,band_LOW_PAIRED__Z2  = 0
 c4254: - aux_c_b6,r5 + sv_function_helpervar_b6,r5__condX1
        + sv_function_helpervar_b6,r5__condX3
        + sv_function_helpervar_b6,r5__condX2  = 0
 c4255: - aux_cap_helper__b8,r8,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b8,r8,band_HIGH_PAIRED__condX1 <= 0
 c4256: - 0.6181448971 aux_cap_helper__b7,r4,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b7,r4,band_HIGH_PAIRED__Z3
        + 1.5453622427 aux_cap_helper__b7,r4,band_HIGH_PAIRED__condX1
        + 2.1635071398 aux_cap_helper__b7,r4,band_HIGH_PAIRED__condX2
        + 1.3908260184 aux_cap_helper__b7,r4,band_HIGH_PAIRED__condX3
        - aux_cap_b7,r4,band_HIGH_PAIRED  = 0
 c4257: - aux_cap_b0,r1,band_LOW_PAIRED
        + 5.2303475434 aux_cap_helper__b0,r1,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b0,r1,band_LOW_PAIRED__condX1
        - 1.4943850124 aux_cap_helper__b0,r1,band_LOW_PAIRED__Z2  = 0
 c4258: - aux_cap_helper__b3,r4,band_LOW_PAIRED__condX1 <= 0
 c4259: - 1.4943850124 aux_cap_helper__b6,r2,band_LOW_PAIRED__Z2
        + 5.2303475434 aux_cap_helper__b6,r2,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b6,r2,band_LOW_PAIRED__condX1
        - aux_cap_b6,r2,band_LOW_PAIRED  = 0
 c4260: aux_cap_helper__b0,r4,band_UNPAIRED__condX1
        + aux_cap_helper__b0,r4,band_UNPAIRED__condX2 - X__b0,r4,band_UNPAIRED
         = 0
 c4261: sv_function_helpervar_b2,r2__condX3
        - 15.9946187432626 sv_function_helpervar_b2,r2__Z3 <= 0
 c4262: aux_cap_helper__b3,r3,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b3,r3,band_UNPAIRED__Z2 <= 0
 c4263: GlobalBidder_PSI__i9,k0 - 18497526.8940961 GlobalBidder_hat{W}__i9,k0
        <= 0
 c4264: aux_cap_helper__b3,r10,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b3,r10,band_HIGH_PAIRED__Z2 <= 0
 c4265: aux_cap_helper__b3,r6,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b3,r6,band_UNPAIRED__Z2 <= 0
 c4266: sv_function_helpervar_b7,r6__Z2 + sv_function_helpervar_b7,r6__Z3
        + sv_function_helpervar_b7,r6__Z1  = 1
 c4267: 7.72501854783287 sv_function_helpervar_b7,r0__Z3
        - sv_function_helpervar_b7,r0__condX3 <= 0
 c4268: - sv_function_helpervar_b6,r12__condX2
        + 0.15994618744279 sv_function_helpervar_b6,r12__Z2 <= 0
 c4269: aux_cap_helper__b4,r2,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b4,r2,band_UNPAIRED__Z2 <= 0
 c4270: - 0.6181448971 aux_cap_helper__b6,r8,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b6,r8,band_HIGH_PAIRED__Z3
        + 1.3908260184 aux_cap_helper__b6,r8,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b6,r8,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b6,r8,band_HIGH_PAIRED__condX1
        - aux_cap_b6,r8,band_HIGH_PAIRED  = 0
 c4271: - 3 aux_cap_helper__b0,r1,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b0,r1,band_HIGH_PAIRED__condX3 <= 0
 c4272: - aux_cap_helper__b3,r4,band_LOW_PAIRED__condX2
        + aux_cap_helper__b3,r4,band_LOW_PAIRED__Z2 <= 0
 c4273: - 3 aux_cap_helper__b5,r10,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b5,r10,band_HIGH_PAIRED__condX3 <= 0
 c4274: - 0.6181448971 aux_cap_helper__b9,r5,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b9,r5,band_HIGH_PAIRED__Z3
        - aux_cap_b9,r5,band_HIGH_PAIRED
        + 1.5453622427 aux_cap_helper__b9,r5,band_HIGH_PAIRED__condX1
        + 1.3908260184 aux_cap_helper__b9,r5,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b9,r5,band_HIGH_PAIRED__condX2  = 0
 c4275: 1.1250244065 aux_cap_helper__b2,r4,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b2,r4,band_UNPAIRED__condX1
        - 0.3214355447 aux_cap_helper__b2,r4,band_UNPAIRED__Z2
        - aux_cap_b2,r4,band_UNPAIRED  = 0
 c4276: 0.8035888618 aux_cap_helper__b5,r6,band_UNPAIRED__condX1
        + 1.1250244065 aux_cap_helper__b5,r6,band_UNPAIRED__condX2
        - aux_cap_b5,r6,band_UNPAIRED
        - 0.3214355447 aux_cap_helper__b5,r6,band_UNPAIRED__Z2  = 0
 c4277: 0.0102040816326531 X__b9,r2,band_UNPAIRED
        + 0.0102040816326531 X__b9,r2,band_HIGH_PAIRED
        - GlobalBidder_W_ir__b9,r2
        + 0.0102040816326531 X__b9,r2,band_LOW_PAIRED <= 0
 c4278: - aux_cap_helper__b0,r12,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b0,r12,band_HIGH_PAIRED__condX1 <= 0
 c4279: aux_cap_helper__b3,r12,band_LOW_PAIRED__condX1
        + aux_cap_helper__b3,r12,band_LOW_PAIRED__condX2
        - X__b3,r12,band_LOW_PAIRED  = 0
 c4280: - 2 aux_cap_helper__b8,r2,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b8,r2,band_HIGH_PAIRED__condX2 <= 0
 c4281: aux_cap_helper__b9,r12,band_LOW_PAIRED__condX1
        - aux_cap_helper__b9,r12,band_LOW_PAIRED__Z1 <= 0
 c4282: - sv_function_helpervar_b3,r8__condX1 <= 0
 c4283: - aux_cap_helper__b6,r5,band_LOW_PAIRED__condX2
        + aux_cap_helper__b6,r5,band_LOW_PAIRED__Z2 <= 0
 c4284: - 2 aux_cap_helper__b2,r7,band_UNPAIRED__Z2
        + aux_cap_helper__b2,r7,band_UNPAIRED__condX2 <= 0
 c4285: 5.65845556019633 sv_function_helpervar_b0,r10__Z3
        - sv_function_helpervar_b0,r10__condX3 <= 0
 c4286: sv_function_helpervar_b1,r2__condX1
        - 0.159946187459287 sv_function_helpervar_b1,r2__Z1 <= 0
 c4287: - 3 aux_cap_helper__b1,r10,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b1,r10,band_HIGH_PAIRED__condX3 <= 0
 c4288: sv_function_helpervar_b5,r10__condX2
        - 5.86499039658243 sv_function_helpervar_b5,r10__Z2 <= 0
 c4289: - aux_c_b7,r12 + sv_function_helpervar_b7,r12__condX3
        + sv_function_helpervar_b7,r12__condX2
        + sv_function_helpervar_b7,r12__condX1  = 0
 c4290: - aux_cap_b4,r1,band_HIGH_PAIRED
        + 1.5453622427 aux_cap_helper__b4,r1,band_HIGH_PAIRED__condX1
        + 2.1635071398 aux_cap_helper__b4,r1,band_HIGH_PAIRED__condX2
        + 1.3908260184 aux_cap_helper__b4,r1,band_HIGH_PAIRED__condX3
        + 0.927217345653617 aux_cap_helper__b4,r1,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b4,r1,band_HIGH_PAIRED__Z2  = 0
 c4291: - 0.3214355447 aux_cap_helper__b6,r1,band_UNPAIRED__Z2
        - aux_cap_b6,r1,band_UNPAIRED
        + 0.8035888618 aux_cap_helper__b6,r1,band_UNPAIRED__condX1
        + 1.1250244065 aux_cap_helper__b6,r1,band_UNPAIRED__condX2  = 0
 c4292: aux_cap_helper__b6,r12,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b6,r12,band_HIGH_PAIRED__Z3 <= 0
 c4293: aux_cap_helper__b9,r6,band_UNPAIRED__condX1
        - aux_cap_helper__b9,r6,band_UNPAIRED__Z1 <= 0
 c4294: aux_cap_helper__b0,r7,band_LOW_PAIRED__Z1
        + aux_cap_helper__b0,r7,band_LOW_PAIRED__Z2  = 1
 c4295: - aux_cap_helper__b2,r10,band_LOW_PAIRED__Z1
        + aux_cap_helper__b2,r10,band_LOW_PAIRED__condX1 <= 0
 c4296: - 2 aux_cap_helper__b3,r11,band_UNPAIRED__Z2
        + aux_cap_helper__b3,r11,band_UNPAIRED__condX2 <= 0
 c4297: aux_cap_helper__b4,r9,band_UNPAIRED__Z2
        - aux_cap_helper__b4,r9,band_UNPAIRED__condX2 <= 0
 c4298: aux_cap_helper__b5,r2,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b5,r2,band_HIGH_PAIRED__condX2 <= 0
 c4299: aux_cap_helper__b5,r10,band_LOW_PAIRED__Z1
        + aux_cap_helper__b5,r10,band_LOW_PAIRED__Z2  = 1
 c4300: - aux_cap_helper__b4,r11,band_LOW_PAIRED__Z1
        + aux_cap_helper__b4,r11,band_LOW_PAIRED__condX1 <= 0
 c4301: - aux_cap_helper__b6,r8,band_UNPAIRED__condX2
        + aux_cap_helper__b6,r8,band_UNPAIRED__Z2 <= 0
 c4302: - 1.4943850124 aux_cap_helper__b2,r9,band_LOW_PAIRED__Z2
        - aux_cap_b2,r9,band_LOW_PAIRED
        + 3.735962531 aux_cap_helper__b2,r9,band_LOW_PAIRED__condX1
        + 5.2303475434 aux_cap_helper__b2,r9,band_LOW_PAIRED__condX2  = 0
 c4303: - 2 aux_cap_helper__b4,r6,band_LOW_PAIRED__Z2
        + aux_cap_helper__b4,r6,band_LOW_PAIRED__condX2 <= 0
 c4304: 5.2303475434 aux_cap_helper__b7,r10,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b7,r10,band_LOW_PAIRED__condX1
        - 1.4943850124 aux_cap_helper__b7,r10,band_LOW_PAIRED__Z2
        - aux_cap_b7,r10,band_LOW_PAIRED  = 0
 c4305: aux_cap_helper__b3,r10,band_LOW_PAIRED__condX2
        + aux_cap_helper__b3,r10,band_LOW_PAIRED__condX1
        - X__b3,r10,band_LOW_PAIRED  = 0
 c4306: - 0.6181448971 aux_cap_helper__b5,r7,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b5,r7,band_HIGH_PAIRED__Z3
        + 2.1635071398 aux_cap_helper__b5,r7,band_HIGH_PAIRED__condX2
        + 1.3908260184 aux_cap_helper__b5,r7,band_HIGH_PAIRED__condX3
        + 1.5453622427 aux_cap_helper__b5,r7,band_HIGH_PAIRED__condX1
        - aux_cap_b5,r7,band_HIGH_PAIRED  = 0
 c4307: aux_cap_helper__b7,r0,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b7,r0,band_HIGH_PAIRED__condX2 <= 0
 c4308: 6.55357470302108 sv_function_helpervar_b8,r5__Z3
        - sv_function_helpervar_b8,r5__condX3 <= 0
 c4309: - aux_cap_helper__b4,r9,band_UNPAIRED__condX1 <= 0
 c4310: - aux_cap_helper__b5,r2,band_HIGH_PAIRED__condX1 <= 0
 c4311: 1.1250244065 aux_cap_helper__b5,r4,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b5,r4,band_UNPAIRED__condX1
        - aux_cap_b5,r4,band_UNPAIRED
        - 0.3214355447 aux_cap_helper__b5,r4,band_UNPAIRED__Z2  = 0
 c4312: X__b2,r9,band_HIGH_PAIRED + X__b6,r9,band_HIGH_PAIRED
        + X__b5,r9,band_HIGH_PAIRED + X__b8,r9,band_HIGH_PAIRED
        + X__b9,r9,band_HIGH_PAIRED + X__b4,r9,band_HIGH_PAIRED
        + X__b0,r9,band_HIGH_PAIRED + X__b7,r9,band_HIGH_PAIRED
        + X__b3,r9,band_HIGH_PAIRED + X__b1,r9,band_HIGH_PAIRED <= 3
 c4313: sv_function_helpervar_b6,r7__condX1
        + sv_function_helpervar_b6,r7__condX2
        + sv_function_helpervar_b6,r7__condX3 - aux_c_b6,r7  = 0
 c4314: - X__b9,r10,band_UNPAIRED
        + aux_cap_helper__b9,r10,band_UNPAIRED__condX2
        + aux_cap_helper__b9,r10,band_UNPAIRED__condX1  = 0
 c4315: - aux_cap_helper__b2,r2,band_HIGH_PAIRED__condX1 <= 0
 c4316: - aux_cap_helper__b7,r0,band_HIGH_PAIRED__condX1 <= 0
 c4317: - aux_c_b8,r4 + sv_function_helpervar_b8,r4__condX3
        + sv_function_helpervar_b8,r4__condX1
        + sv_function_helpervar_b8,r4__condX2  = 0
 c4318: aux_cap_helper__b5,r12,band_UNPAIRED__Z2
        - aux_cap_helper__b5,r12,band_UNPAIRED__condX2 <= 0
 c4319: - aux_cap_helper__b5,r8,band_LOW_PAIRED__condX1 <= 0
 c4320: - aux_cap_helper__b5,r12,band_UNPAIRED__condX1 <= 0
 c4321: - sv_function_helpervar_b2,r5__condX1 <= 0
 c4322: - aux_cap_helper__b7,r13,band_UNPAIRED__condX2
        + aux_cap_helper__b7,r13,band_UNPAIRED__Z2 <= 0
 c4323: - aux_Omega_b5,r8 + 231.284798959184 aux_sv_b5,r8  = 0
 c4324: - aux_cap_helper__b5,r5,band_UNPAIRED__condX2
        + aux_cap_helper__b5,r5,band_UNPAIRED__Z2 <= 0
 c4325: - 0.3214355447 aux_cap_helper__b3,r1,band_UNPAIRED__Z2
        + 1.1250244065 aux_cap_helper__b3,r1,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b3,r1,band_UNPAIRED__condX1
        - aux_cap_b3,r1,band_UNPAIRED  = 0
 c4326: sv_function_helpervar_b4,r9__condX1
        - 0.159946187437935 sv_function_helpervar_b4,r9__Z1 <= 0
 c4327: aux_cap_helper__b5,r10,band_LOW_PAIRED__condX1
        - aux_cap_helper__b5,r10,band_LOW_PAIRED__Z1 <= 0
 c4328: - 6.3832954915594 sv_function_helpervar_b5,r13__Z2
        + sv_function_helpervar_b5,r13__condX2 <= 0
 c4329: aux_cap_b8,r10,band_HIGH_PAIRED - aux_c_b8,r10
        + aux_cap_b8,r10,band_UNPAIRED + aux_cap_b8,r10,band_LOW_PAIRED  = 0
 c4330: aux_cap_helper__b0,r13,band_LOW_PAIRED__Z1
        + aux_cap_helper__b0,r13,band_LOW_PAIRED__Z2  = 1
 c4331: - 2 aux_cap_helper__b4,r12,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b4,r12,band_HIGH_PAIRED__condX2 <= 0
 c4332: aux_cap_helper__b7,r11,band_LOW_PAIRED__condX1
        - aux_cap_helper__b7,r11,band_LOW_PAIRED__Z1 <= 0
 c4333: - sv_function_helpervar_b7,r5__condX1 <= 0
 c4334: aux_cap_helper__b4,r0,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b4,r0,band_HIGH_PAIRED__Z1 <= 0
 c4335: 5.2303475434 aux_cap_helper__b7,r5,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b7,r5,band_LOW_PAIRED__condX1
        - 1.4943850124 aux_cap_helper__b7,r5,band_LOW_PAIRED__Z2
        - aux_cap_b7,r5,band_LOW_PAIRED  = 0
 c4336: aux_cap_helper__b9,r12,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b9,r12,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b9,r12,band_HIGH_PAIRED__Z1  = 1
 c4337: aux_cap_b6,r1,band_LOW_PAIRED + aux_cap_b6,r1,band_HIGH_PAIRED
        - aux_c_b6,r1 + aux_cap_b6,r1,band_UNPAIRED  = 0
 c4338: aux_cap_b5,r3,band_HIGH_PAIRED - aux_c_b5,r3
        + aux_cap_b5,r3,band_LOW_PAIRED + aux_cap_b5,r3,band_UNPAIRED  = 0
 c4339: aux_cap_helper__b8,r9,band_LOW_PAIRED__Z2
        - aux_cap_helper__b8,r9,band_LOW_PAIRED__condX2 <= 0
 c4340: aux_cap_helper__b4,r8,band_UNPAIRED__Z1
        + aux_cap_helper__b4,r8,band_UNPAIRED__Z2  = 1
 c4341: - 2 aux_cap_helper__b0,r8,band_UNPAIRED__Z2
        + aux_cap_helper__b0,r8,band_UNPAIRED__condX2 <= 0
 c4342: sv_function_helpervar_b8,r8__condX1
        - 0.159946187496666 sv_function_helpervar_b8,r8__Z1 <= 0
 c4343: - aux_cap_helper__b1,r13,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b1,r13,band_HIGH_PAIRED__Z3 <= 0
 c4344: aux_cap_helper__b0,r11,band_UNPAIRED__Z2
        - aux_cap_helper__b0,r11,band_UNPAIRED__condX2 <= 0
 c4345: 27.9544515927 sv_function_helpervar_b8,r2__condX3
        + 44.9021177059 sv_function_helpervar_b8,r2__condX2
        + 1343.0085755364 sv_function_helpervar_b8,r2__condX1 - aux_sv_b8,r2
        + 348.46846886708 sv_function_helpervar_b8,r2__Z3
        + 207.627178817628 sv_function_helpervar_b8,r2__Z2  = 0
 c4346: aux_cap_helper__b5,r8,band_LOW_PAIRED__Z2
        - aux_cap_helper__b5,r8,band_LOW_PAIRED__condX2 <= 0
 c4347: sv_function_helpervar_b5,r12__Z2 + sv_function_helpervar_b5,r12__Z1
        + sv_function_helpervar_b5,r12__Z3  = 1
 c4348: - aux_Omega_b7,r10 + 745.318901567386 aux_sv_b7,r10  = 0
 c4349: - aux_cap_helper__b8,r9,band_LOW_PAIRED__condX1 <= 0
 c4350: 1169.5974422857 sv_function_helpervar_b5,r8__condX1
        + 19.4616087373 sv_function_helpervar_b5,r8__condX3
        + 51.2218102438 sv_function_helpervar_b5,r8__condX2 - aux_sv_b5,r8
        + 178.879918485976 sv_function_helpervar_b5,r8__Z2
        + 381.580661261441 sv_function_helpervar_b5,r8__Z3  = 0
 c4351: aux_cap_helper__b1,r13,band_LOW_PAIRED__condX1
        + aux_cap_helper__b1,r13,band_LOW_PAIRED__condX2
        - X__b1,r13,band_LOW_PAIRED  = 0
 c4352: - 2 aux_cap_helper__b6,r1,band_UNPAIRED__Z2
        + aux_cap_helper__b6,r1,band_UNPAIRED__condX2 <= 0
 c4353: - sv_function_helpervar_b5,r1__condX2
        + 0.159946187453604 sv_function_helpervar_b5,r1__Z2 <= 0
 c4354: sv_function_helpervar_b5,r0__condX3
        - 15.9946187432626 sv_function_helpervar_b5,r0__Z3 <= 0
 c4355: - 1.4943850124 aux_cap_helper__b0,r2,band_LOW_PAIRED__Z2
        - aux_cap_b0,r2,band_LOW_PAIRED
        + 5.2303475434 aux_cap_helper__b0,r2,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b0,r2,band_LOW_PAIRED__condX1  = 0
 c4356: - 0.159946187484381 sv_function_helpervar_b3,r11__Z1
        + sv_function_helpervar_b3,r11__condX1 <= 0
 c4357: aux_cap_helper__b2,r7,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b2,r7,band_LOW_PAIRED__Z2 <= 0
 c4358: - aux_cap_helper__b1,r1,band_UNPAIRED__Z1
        + aux_cap_helper__b1,r1,band_UNPAIRED__condX1 <= 0
 c4359: aux_cap_helper__b7,r9,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b7,r9,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b7,r9,band_HIGH_PAIRED__Z3  = 1
 c4360: sv_function_helpervar_b2,r3__condX3
        - 15.9946187432626 sv_function_helpervar_b2,r3__Z3 <= 0
 c4361: - aux_cap_helper__b9,r13,band_UNPAIRED__condX1 <= 0
 c4362: aux_cap_helper__b6,r6,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b6,r6,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b6,r6,band_HIGH_PAIRED__Z2  = 1
 c4363: - sv_function_helpervar_b7,r4__condX1 <= 0
 c4364: aux_cap_helper__b0,r10,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b0,r10,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b0,r10,band_HIGH_PAIRED__Z3  = 1
 c4365: 41.0673172511379 aux_sv_b0,r6 - aux_Omega_b0,r6  = 0
 c4366: aux_cap_helper__b1,r5,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b1,r5,band_HIGH_PAIRED__Z2 <= 0
 c4367: sv_function_helpervar_b6,r2__condX2
        - 5.52779119010926 sv_function_helpervar_b6,r2__Z2 <= 0
 c4368: aux_cap_helper__b7,r13,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b7,r13,band_LOW_PAIRED__Z2 <= 0
 c4369: 0.159946187436161 sv_function_helpervar_b3,r1__Z2
        - sv_function_helpervar_b3,r1__condX2 <= 0
 c4370: aux_cap_helper__b0,r9,band_UNPAIRED__Z2
        - aux_cap_helper__b0,r9,band_UNPAIRED__condX2 <= 0
 c4371: aux_cap_helper__b2,r8,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b2,r8,band_HIGH_PAIRED__condX2 <= 0
 c4372: sv_function_helpervar_b8,r8__condX3
        - 15.9946187432626 sv_function_helpervar_b8,r8__Z3 <= 0
 c4373: - sv_function_helpervar_b5,r5__condX3
        + 6.26596121586556 sv_function_helpervar_b5,r5__Z3 <= 0
 c4374: - 5.94064266021643 sv_function_helpervar_b5,r3__Z2
        + sv_function_helpervar_b5,r3__condX2 <= 0
 c4375: sv_function_helpervar_b2,r1__condX1
        - 0.159946187435238 sv_function_helpervar_b2,r1__Z1 <= 0
 c4376: - aux_cap_b0,r9,band_LOW_PAIRED
        + 3.735962531 aux_cap_helper__b0,r9,band_LOW_PAIRED__condX1
        + 5.2303475434 aux_cap_helper__b0,r9,band_LOW_PAIRED__condX2
        - 1.4943850124 aux_cap_helper__b0,r9,band_LOW_PAIRED__Z2  = 0
 c4377: aux_cap_helper__b5,r8,band_UNPAIRED__condX1
        - aux_cap_helper__b5,r8,band_UNPAIRED__Z1 <= 0
 c4378: sv_function_helpervar_b9,r3__Z3 + sv_function_helpervar_b9,r3__Z1
        + sv_function_helpervar_b9,r3__Z2  = 1
 c4379: - aux_cap_helper__b0,r9,band_UNPAIRED__condX1 <= 0
 c4380: - aux_Omega_b1,r1 + 194.218278054677 aux_sv_b1,r1  = 0
 c4381: - sv_function_helpervar_b9,r4__condX3
        + 6.52508471074699 sv_function_helpervar_b9,r4__Z3 <= 0
 c4382: aux_cap_b1,r13,band_HIGH_PAIRED + aux_cap_b1,r13,band_LOW_PAIRED
        - aux_c_b1,r13 + aux_cap_b1,r13,band_UNPAIRED  = 0
 c4383: - 18497526.8940961 GlobalBidder_hat{W}__i8,k2 + GlobalBidder_PSI__i8,k2
        <= 0
 c4384: aux_cap_helper__b0,r11,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b0,r11,band_HIGH_PAIRED__Z2 <= 0
 c4385: - aux_cap_helper__b4,r3,band_UNPAIRED__Z1
        + aux_cap_helper__b4,r3,band_UNPAIRED__condX1 <= 0
 c4386: - aux_sv_b4,r11 + 437.341737165636 sv_function_helpervar_b4,r11__Z3
        + 196.416617814964 sv_function_helpervar_b4,r11__Z2
        + 61.3103850893 sv_function_helpervar_b4,r11__condX2
        + 1289.3272634025 sv_function_helpervar_b4,r11__condX1
        + 20.4098067911 sv_function_helpervar_b4,r11__condX3  = 0
 c4387: sv_function_helpervar_b4,r0__condX2
        + sv_function_helpervar_b4,r0__condX1
        + sv_function_helpervar_b4,r0__condX3 - aux_c_b4,r0  = 0
 c4388: - aux_cap_helper__b9,r4,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b9,r4,band_HIGH_PAIRED__Z3 <= 0
 c4389: - 2 aux_cap_helper__b0,r13,band_UNPAIRED__Z2
        + aux_cap_helper__b0,r13,band_UNPAIRED__condX2 <= 0
 c4390: aux_cap_helper__b4,r1,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b4,r1,band_HIGH_PAIRED__Z2 <= 0
 c4391: - aux_cap_helper__b9,r13,band_UNPAIRED__condX2
        + aux_cap_helper__b9,r13,band_UNPAIRED__Z2 <= 0
 c4392: - aux_c_b7,r1 + sv_function_helpervar_b7,r1__condX2
        + sv_function_helpervar_b7,r1__condX1
        + sv_function_helpervar_b7,r1__condX3  = 0
 c4393: aux_cap_helper__b1,r11,band_LOW_PAIRED__condX1
        - aux_cap_helper__b1,r11,band_LOW_PAIRED__Z1 <= 0
 c4394: - sv_function_helpervar_b1,r13__condX1 <= 0
 c4395: sv_function_helpervar_b5,r10__condX1
        - 0.159946187496824 sv_function_helpervar_b5,r10__Z1 <= 0
 c4396: - aux_c_b7,r13 + sv_function_helpervar_b7,r13__condX3
        + sv_function_helpervar_b7,r13__condX2
        + sv_function_helpervar_b7,r13__condX1  = 0
 c4397: - aux_cap_helper__b6,r13,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b6,r13,band_HIGH_PAIRED__Z2 <= 0
 c4398: - aux_Omega_b9,r3 + 1045.55692293268 aux_sv_b9,r3  = 0
 c4399: - 18497526.8940961 GlobalBidder_hat{W}__i8,k0 - GlobalBidder_PSI__i8,k0
        + aux_Omega_b8,r12 + aux_Omega_b8,r13 + aux_Omega_b8,r10
        + aux_Omega_b8,r11 + aux_Omega_b8,r5 + aux_Omega_b8,r6
        + aux_Omega_b8,r7 + aux_Omega_b8,r8 + aux_Omega_b8,r9 + aux_Omega_b8,r0
        + aux_Omega_b8,r1 + aux_Omega_b8,r2 + aux_Omega_b8,r3 + aux_Omega_b8,r4
        >= -18497526.8940961
 c4400: - 2 aux_cap_helper__b6,r7,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b6,r7,band_HIGH_PAIRED__condX2 <= 0
 c4401: aux_cap_helper__b9,r8,band_LOW_PAIRED__Z2
        + aux_cap_helper__b9,r8,band_LOW_PAIRED__Z1  = 1
 c4402: aux_cap_helper__b9,r7,band_UNPAIRED__condX2
        + aux_cap_helper__b9,r7,band_UNPAIRED__condX1 - X__b9,r7,band_UNPAIRED
         = 0
 c4403: aux_cap_helper__b0,r3,band_LOW_PAIRED__Z1
        + aux_cap_helper__b0,r3,band_LOW_PAIRED__Z2  = 1
 c4404: - aux_c_b2,r9 + sv_function_helpervar_b2,r9__condX1
        + sv_function_helpervar_b2,r9__condX2
        + sv_function_helpervar_b2,r9__condX3  = 0
 c4405: - aux_cap_helper__b3,r12,band_UNPAIRED__condX1 <= 0
 c4406: sv_function_helpervar_b0,r2__condX3
        + sv_function_helpervar_b0,r2__condX1
        + sv_function_helpervar_b0,r2__condX2 - aux_c_b0,r2  = 0
 c4407: aux_cap_helper__b0,r12,band_LOW_PAIRED__condX1
        - aux_cap_helper__b0,r12,band_LOW_PAIRED__Z1 <= 0
 c4408: X__b7,r0,band_UNPAIRED - GlobalBidder_W_ir__b7,r0
        + X__b7,r0,band_HIGH_PAIRED + X__b7,r0,band_LOW_PAIRED >= 0
 c4409: - X__b1,r10,band_HIGH_PAIRED
        + aux_cap_helper__b1,r10,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b1,r10,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b1,r10,band_HIGH_PAIRED__condX1  = 0
 c4410: 5.2303475434 aux_cap_helper__b9,r5,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b9,r5,band_LOW_PAIRED__condX1
        - 1.4943850124 aux_cap_helper__b9,r5,band_LOW_PAIRED__Z2
        - aux_cap_b9,r5,band_LOW_PAIRED  = 0
 c4411: sv_function_helpervar_b6,r6__condX2
        - 5.80305813698008 sv_function_helpervar_b6,r6__Z2 <= 0
 c4412: sv_function_helpervar_b3,r3__condX3
        + sv_function_helpervar_b3,r3__condX2
        + sv_function_helpervar_b3,r3__condX1 - aux_c_b3,r3  = 0
 c4413: - aux_cap_helper__b2,r8,band_HIGH_PAIRED__condX1 <= 0
 c4414: - aux_cap_helper__b7,r5,band_UNPAIRED__Z1
        + aux_cap_helper__b7,r5,band_UNPAIRED__condX1 <= 0
 c4415: 32.5515297557 sv_function_helpervar_b7,r11__condX3
        + 74.7775024056 sv_function_helpervar_b7,r11__condX2
        + 1850.3233259353 sv_function_helpervar_b7,r11__condX1 - aux_sv_b7,r11
        + 575.469808750544 sv_function_helpervar_b7,r11__Z3
        + 283.991785093775 sv_function_helpervar_b7,r11__Z2  = 0
 c4416: - sv_function_helpervar_b9,r0__condX3
        + 7.36804854257395 sv_function_helpervar_b9,r0__Z3 <= 0
 c4417: - X__b3,r9,band_LOW_PAIRED
        + aux_cap_helper__b3,r9,band_LOW_PAIRED__condX2
        + aux_cap_helper__b3,r9,band_LOW_PAIRED__condX1  = 0
 c4418: aux_cap_helper__b5,r10,band_UNPAIRED__Z2
        - aux_cap_helper__b5,r10,band_UNPAIRED__condX2 <= 0
 c4419: - 3 aux_cap_helper__b6,r1,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b6,r1,band_HIGH_PAIRED__condX3 <= 0
 c4420: - 2 aux_cap_helper__b1,r8,band_LOW_PAIRED__Z2
        + aux_cap_helper__b1,r8,band_LOW_PAIRED__condX2 <= 0
 c4421: 0.159946187436986 sv_function_helpervar_b4,r1__Z2
        - sv_function_helpervar_b4,r1__condX2 <= 0
 c4422: - 3 aux_cap_helper__b8,r7,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b8,r7,band_HIGH_PAIRED__condX3 <= 0
 c4423: aux_cap_helper__b8,r4,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b8,r4,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b8,r4,band_HIGH_PAIRED__Z1  = 1
 c4424: - X__b8,r7,band_LOW_PAIRED
        + aux_cap_helper__b8,r7,band_LOW_PAIRED__condX2
        + aux_cap_helper__b8,r7,band_LOW_PAIRED__condX1  = 0
 c4425: aux_cap_helper__b2,r2,band_UNPAIRED__condX1
        + aux_cap_helper__b2,r2,band_UNPAIRED__condX2 - X__b2,r2,band_UNPAIRED
         = 0
 c4426: aux_cap_helper__b5,r5,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b5,r5,band_HIGH_PAIRED__condX2 <= 0
 c4427: 3.735962531 aux_cap_helper__b3,r0,band_LOW_PAIRED__condX1
        + 5.2303475434 aux_cap_helper__b3,r0,band_LOW_PAIRED__condX2
        - 1.4943850124 aux_cap_helper__b3,r0,band_LOW_PAIRED__Z2
        - aux_cap_b3,r0,band_LOW_PAIRED  = 0
 c4428: aux_cap_helper__b0,r2,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b0,r2,band_HIGH_PAIRED__Z1 <= 0
 c4429: - aux_cap_helper__b3,r12,band_UNPAIRED__condX2
        + aux_cap_helper__b3,r12,band_UNPAIRED__Z2 <= 0
 c4430: - 0.159946187500414 sv_function_helpervar_b1,r13__Z1
        + sv_function_helpervar_b1,r13__condX1 <= 0
 c4431: - aux_cap_helper__b9,r3,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b9,r3,band_HIGH_PAIRED__Z2 <= 0
 c4432: - aux_cap_helper__b0,r4,band_UNPAIRED__condX2
        + aux_cap_helper__b0,r4,band_UNPAIRED__Z2 <= 0
 c4433: - sv_function_helpervar_b6,r1__condX2
        + 0.159946187477713 sv_function_helpervar_b6,r1__Z2 <= 0
 c4434: - 1.4943850124 aux_cap_helper__b7,r8,band_LOW_PAIRED__Z2
        + 5.2303475434 aux_cap_helper__b7,r8,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b7,r8,band_LOW_PAIRED__condX1
        - aux_cap_b7,r8,band_LOW_PAIRED  = 0
 c4435: aux_cap_b0,r7,band_UNPAIRED + aux_cap_b0,r7,band_LOW_PAIRED
        + aux_cap_b0,r7,band_HIGH_PAIRED - aux_c_b0,r7  = 0
 c4436: 3.735962531 aux_cap_helper__b4,r13,band_LOW_PAIRED__condX1
        + 5.2303475434 aux_cap_helper__b4,r13,band_LOW_PAIRED__condX2
        - 1.4943850124 aux_cap_helper__b4,r13,band_LOW_PAIRED__Z2
        - aux_cap_b4,r13,band_LOW_PAIRED  = 0
 c4437: aux_Omega_b0,r11 - v__0 + aux_Omega_b0,r6 + aux_Omega_b0,r7  = 0
 c4438: aux_cap_helper__b7,r10,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b7,r10,band_UNPAIRED__Z2 <= 0
 c4439: - 2 aux_cap_helper__b8,r2,band_UNPAIRED__Z2
        + aux_cap_helper__b8,r2,band_UNPAIRED__condX2 <= 0
 c4440: 462.466372956449 sv_function_helpervar_b3,r0__Z3
        + 210.44584263774 sv_function_helpervar_b3,r0__Z2
        + 1379.5885615636 sv_function_helpervar_b3,r0__condX1
        + 63.8595280594 sv_function_helpervar_b3,r0__condX2
        + 22.1819997883 sv_function_helpervar_b3,r0__condX3 - aux_sv_b3,r0  = 0
 c4441: 24.4494834637948 aux_sv_b4,r13 - aux_Omega_b4,r13  = 0
 c4442: - 0.3214355447 aux_cap_helper__b9,r3,band_UNPAIRED__Z2
        - aux_cap_b9,r3,band_UNPAIRED
        + 1.1250244065 aux_cap_helper__b9,r3,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b9,r3,band_UNPAIRED__condX1  = 0
 c4443: aux_cap_helper__b3,r5,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b3,r5,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b3,r5,band_HIGH_PAIRED__Z1  = 1
 c4444: - 6.5782911246891 sv_function_helpervar_b2,r13__Z2
        + sv_function_helpervar_b2,r13__condX2 <= 0
 c4445: aux_cap_helper__b9,r13,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b9,r13,band_HIGH_PAIRED__condX2 <= 0
 c4446: - 2 aux_cap_helper__b8,r8,band_LOW_PAIRED__Z2
        + aux_cap_helper__b8,r8,band_LOW_PAIRED__condX2 <= 0
 c4447: aux_cap_helper__b5,r1,band_LOW_PAIRED__Z2
        + aux_cap_helper__b5,r1,band_LOW_PAIRED__Z1  = 1
 c4448: - aux_cap_b1,r2,band_UNPAIRED
        + 1.1250244065 aux_cap_helper__b1,r2,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b1,r2,band_UNPAIRED__condX1
        - 0.3214355447 aux_cap_helper__b1,r2,band_UNPAIRED__Z2  = 0
 c4449: - aux_cap_b7,r6,band_UNPAIRED
        + 1.1250244065 aux_cap_helper__b7,r6,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b7,r6,band_UNPAIRED__condX1
        - 0.3214355447 aux_cap_helper__b7,r6,band_UNPAIRED__Z2  = 0
 c4450: aux_cap_helper__b0,r4,band_UNPAIRED__Z1
        + aux_cap_helper__b0,r4,band_UNPAIRED__Z2  = 1
 c4451: 3.735962531 aux_cap_helper__b7,r1,band_LOW_PAIRED__condX1
        + 5.2303475434 aux_cap_helper__b7,r1,band_LOW_PAIRED__condX2
        - 1.4943850124 aux_cap_helper__b7,r1,band_LOW_PAIRED__Z2
        - aux_cap_b7,r1,band_LOW_PAIRED  = 0
 c4452: X__b9,r1,band_UNPAIRED + X__b1,r1,band_UNPAIRED
        + X__b0,r1,band_UNPAIRED + X__b4,r1,band_UNPAIRED
        + X__b5,r1,band_UNPAIRED + X__b6,r1,band_UNPAIRED
        + X__b2,r1,band_UNPAIRED + X__b8,r1,band_UNPAIRED
        + X__b7,r1,band_UNPAIRED + X__b3,r1,band_UNPAIRED <= 2
 c4453: aux_cap_helper__b3,r1,band_LOW_PAIRED__condX1
        + aux_cap_helper__b3,r1,band_LOW_PAIRED__condX2
        - X__b3,r1,band_LOW_PAIRED  = 0
 c4454: - 0.3214355447 aux_cap_helper__b4,r10,band_UNPAIRED__Z2
        - aux_cap_b4,r10,band_UNPAIRED
        + 1.1250244065 aux_cap_helper__b4,r10,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b4,r10,band_UNPAIRED__condX1  = 0
 c4455: 5.2303475434 aux_cap_helper__b2,r6,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b2,r6,band_LOW_PAIRED__condX1
        - 1.4943850124 aux_cap_helper__b2,r6,band_LOW_PAIRED__Z2
        - aux_cap_b2,r6,band_LOW_PAIRED  = 0
 c4456: - 0.159946187438714 sv_function_helpervar_b0,r10__Z1
        + sv_function_helpervar_b0,r10__condX1 <= 0
 c4457: - X__b4,r5,band_HIGH_PAIRED
        + aux_cap_helper__b4,r5,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b4,r5,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b4,r5,band_HIGH_PAIRED__condX2  = 0
 c4458: - aux_cap_helper__b9,r13,band_HIGH_PAIRED__condX1 <= 0
 c4459: - aux_cap_helper__b5,r3,band_UNPAIRED__condX2
        + aux_cap_helper__b5,r3,band_UNPAIRED__Z2 <= 0
 c4460: sv_function_helpervar_b9,r8__condX3
        + sv_function_helpervar_b9,r8__condX2
        + sv_function_helpervar_b9,r8__condX1 - aux_c_b9,r8  = 0
 c4461: 6.63228196246224 sv_function_helpervar_b2,r6__Z3
        - sv_function_helpervar_b2,r6__condX3 <= 0
 c4462: 629.094716577369 aux_sv_b0,r5 - aux_Omega_b0,r5  = 0
 c4463: 2 aux_cap_helper__b4,r2,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b4,r2,band_HIGH_PAIRED__condX3 <= 0
 c4464: aux_cap_b8,r8,band_LOW_PAIRED - aux_c_b8,r8
        + aux_cap_b8,r8,band_HIGH_PAIRED + aux_cap_b8,r8,band_UNPAIRED  = 0
 c4465: aux_Omega_b7,r10 + aux_Omega_b7,r11 + aux_Omega_b7,r12
        + aux_Omega_b7,r13 + aux_Omega_b7,r0 + aux_Omega_b7,r1
        + aux_Omega_b7,r2 + aux_Omega_b7,r3 + aux_Omega_b7,r4 + aux_Omega_b7,r5
        + aux_Omega_b7,r6 + aux_Omega_b7,r7 + aux_Omega_b7,r8 + aux_Omega_b7,r9
        - GlobalBidder_PSI__i7,k2 + 18497526.8940961 GlobalBidder_hat{W}__i7,k2
        <= 18497526.8940961
 c4466: aux_cap_helper__b2,r0,band_LOW_PAIRED__condX1
        - aux_cap_helper__b2,r0,band_LOW_PAIRED__Z1 <= 0
 c4467: aux_cap_helper__b6,r6,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b6,r6,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b6,r6,band_HIGH_PAIRED__condX1
        - X__b6,r6,band_HIGH_PAIRED  = 0
 c4468: aux_cap_helper__b8,r1,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b8,r1,band_HIGH_PAIRED__Z1 <= 0
 c4469: - aux_cap_helper__b5,r3,band_UNPAIRED__condX1 <= 0
 c4470: - aux_sv_b8,r0 + 22.0614376905 sv_function_helpervar_b8,r0__condX3
        + 1343.0085755585 sv_function_helpervar_b8,r0__condX1
        + 60.0167649891 sv_function_helpervar_b8,r0__condX2
        + 442.724979481219 sv_function_helpervar_b8,r0__Z3
        + 205.20964861038 sv_function_helpervar_b8,r0__Z2  = 0
 c4471: aux_cap_helper__b3,r10,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b3,r10,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b3,r10,band_HIGH_PAIRED__condX3
        - X__b3,r10,band_HIGH_PAIRED  = 0
 c4472: - sv_function_helpervar_b3,r9__condX1 <= 0
 c4473: aux_cap_helper__b0,r9,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b0,r9,band_LOW_PAIRED__Z2 <= 0
 c4474: aux_cap_helper__b4,r9,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b4,r9,band_HIGH_PAIRED__Z3 <= 0
 c4475: - aux_cap_b3,r10,band_LOW_PAIRED
        + 5.2303475434 aux_cap_helper__b3,r10,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b3,r10,band_LOW_PAIRED__condX1
        - 1.4943850124 aux_cap_helper__b3,r10,band_LOW_PAIRED__Z2  = 0
 c4476: aux_cap_b5,r8,band_UNPAIRED - aux_c_b5,r8
        + aux_cap_b5,r8,band_LOW_PAIRED + aux_cap_b5,r8,band_HIGH_PAIRED  = 0
 c4477: sv_function_helpervar_b0,r2__condX2
        - 6.44480188964475 sv_function_helpervar_b0,r2__Z2 <= 0
 c4478: - X__b1,r5,band_HIGH_PAIRED
        + aux_cap_helper__b1,r5,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b1,r5,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b1,r5,band_HIGH_PAIRED__condX1  = 0
 c4479: - aux_cap_helper__b0,r3,band_HIGH_PAIRED__condX1 <= 0
 c4480: - aux_cap_helper__b6,r13,band_HIGH_PAIRED__condX1 <= 0
 c4481: - aux_cap_helper__b6,r6,band_HIGH_PAIRED__condX1 <= 0
 c4482: - aux_cap_helper__b7,r0,band_UNPAIRED__Z1
        + aux_cap_helper__b7,r0,band_UNPAIRED__condX1 <= 0
 c4483: sv_function_helpervar_b2,r2__Z3 + sv_function_helpervar_b2,r2__Z1
        + sv_function_helpervar_b2,r2__Z2  = 1
 c4484: - aux_cap_helper__b9,r3,band_HIGH_PAIRED__condX1 <= 0
 c4485: sv_function_helpervar_b1,r1__Z3 + sv_function_helpervar_b1,r1__Z1
        + sv_function_helpervar_b1,r1__Z2  = 1
 c4486: sv_function_helpervar_b9,r7__Z3 + sv_function_helpervar_b9,r7__Z1
        + sv_function_helpervar_b9,r7__Z2  = 1
 c4487: - X__b3,r6,band_HIGH_PAIRED
        + aux_cap_helper__b3,r6,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b3,r6,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b3,r6,band_HIGH_PAIRED__condX3  = 0
 c4488: 5.2303475434 aux_cap_helper__b5,r12,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b5,r12,band_LOW_PAIRED__condX1
        - aux_cap_b5,r12,band_LOW_PAIRED
        - 1.4943850124 aux_cap_helper__b5,r12,band_LOW_PAIRED__Z2  = 0
 c4489: - aux_sv_b0,r4 + 16.7940083798349 sv_function_helpervar_b0,r4__Z2
        + 36.0806034928766 sv_function_helpervar_b0,r4__Z3
        + 1.8134873934 sv_function_helpervar_b0,r4__condX3
        + 4.8727964575 sv_function_helpervar_b0,r4__condX2
        + 109.8706626013 sv_function_helpervar_b0,r4__condX1  = 0
 c4490: - aux_cap_helper__b3,r11,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b3,r11,band_HIGH_PAIRED__Z3 <= 0
 c4491: 2 aux_cap_helper__b8,r10,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b8,r10,band_HIGH_PAIRED__condX3 <= 0
 c4492: - aux_Omega_b7,r6 + 99.2781472032949 aux_sv_b7,r6  = 0
 c4493: - X__b5,r12,band_HIGH_PAIRED
        + aux_cap_helper__b5,r12,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b5,r12,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b5,r12,band_HIGH_PAIRED__condX1  = 0
 c4494: aux_cap_helper__b1,r12,band_UNPAIRED__condX1
        - aux_cap_helper__b1,r12,band_UNPAIRED__Z1 <= 0
 c4495: - aux_cap_helper__b0,r3,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b0,r3,band_HIGH_PAIRED__Z2 <= 0
 c4496: - aux_cap_helper__b5,r4,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b5,r4,band_HIGH_PAIRED__condX1 <= 0
 c4497: - X__b7,r5,band_UNPAIRED + aux_cap_helper__b7,r5,band_UNPAIRED__condX1
        + aux_cap_helper__b7,r5,band_UNPAIRED__condX2  = 0
 c4498: - aux_c_b8,r7 + aux_cap_b8,r7,band_HIGH_PAIRED
        + aux_cap_b8,r7,band_UNPAIRED + aux_cap_b8,r7,band_LOW_PAIRED  = 0
 c4499: sv_function_helpervar_b6,r6__condX3
        - 15.9946187432626 sv_function_helpervar_b6,r6__Z3 <= 0
 c4500: sv_function_helpervar_b7,r2__condX1
        - 0.159946187532419 sv_function_helpervar_b7,r2__Z1 <= 0
 c4501: - 0.159946187434125 sv_function_helpervar_b1,r10__Z1
        + sv_function_helpervar_b1,r10__condX1 <= 0
 c4502: - aux_cap_helper__b9,r2,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b9,r2,band_HIGH_PAIRED__condX1 <= 0
 c4503: 2 aux_cap_helper__b5,r3,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b5,r3,band_HIGH_PAIRED__condX3 <= 0
 c4504: aux_cap_helper__b2,r11,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b2,r11,band_UNPAIRED__Z2 <= 0
 c4505: sv_function_helpervar_b8,r7__condX2
        - 8.24676410539167 sv_function_helpervar_b8,r7__Z2 <= 0
 c4506: sv_function_helpervar_b4,r5__condX3
        - 15.9946187432626 sv_function_helpervar_b4,r5__Z3 <= 0
 c4507: X__b5,r2,band_UNPAIRED + X__b4,r2,band_UNPAIRED
        + X__b3,r2,band_UNPAIRED + X__b6,r2,band_UNPAIRED
        + X__b1,r2,band_UNPAIRED + X__b9,r2,band_UNPAIRED
        + X__b0,r2,band_UNPAIRED + X__b8,r2,band_UNPAIRED
        + X__b2,r2,band_UNPAIRED + X__b7,r2,band_UNPAIRED <= 2
 c4508: aux_cap_helper__b2,r4,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b2,r4,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b2,r4,band_HIGH_PAIRED__condX3
        - X__b2,r4,band_HIGH_PAIRED  = 0
 c4509: aux_cap_helper__b9,r12,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b9,r12,band_HIGH_PAIRED__Z1 <= 0
 c4510: - 3 aux_cap_helper__b2,r0,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b2,r0,band_HIGH_PAIRED__condX3 <= 0
 c4511: 209.972051865084 sv_function_helpervar_b3,r12__Z2
        + 471.536060749821 sv_function_helpervar_b3,r12__Z3
        + 21.6149535874 sv_function_helpervar_b3,r12__condX3
        + 66.8217166621 sv_function_helpervar_b3,r12__condX2
        + 1379.5885616765 sv_function_helpervar_b3,r12__condX1 - aux_sv_b3,r12
         = 0
 c4512: - aux_cap_helper__b9,r5,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b9,r5,band_HIGH_PAIRED__condX1 <= 0
 c4513: 5.89346615134898 sv_function_helpervar_b2,r9__Z3
        - sv_function_helpervar_b2,r9__condX3 <= 0
 c4514: - aux_cap_helper__b4,r0,band_UNPAIRED__condX1 <= 0
 c4515: aux_cap_helper__b1,r9,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b1,r9,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b1,r9,band_HIGH_PAIRED__Z3  = 1
 c4516: aux_cap_helper__b0,r6,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b0,r6,band_LOW_PAIRED__Z2 <= 0
 c4517: sv_function_helpervar_b8,r10__condX3
        - 15.9946187432626 sv_function_helpervar_b8,r10__Z3 <= 0
 c4518: 681.850220673666 aux_sv_b2,r10 - aux_Omega_b2,r10  = 0
 c4519: 0.159946187436967 sv_function_helpervar_b8,r9__Z2
        - sv_function_helpervar_b8,r9__condX2 <= 0
 c4520: aux_cap_helper__b9,r11,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b9,r11,band_LOW_PAIRED__Z2 <= 0
 c4521: - sv_function_helpervar_b3,r1__condX1 <= 0
 c4522: aux_Omega_b7,r10 + aux_Omega_b7,r11 + aux_Omega_b7,r12
        + aux_Omega_b7,r13 + aux_Omega_b7,r0 + aux_Omega_b7,r1
        + aux_Omega_b7,r2 + aux_Omega_b7,r3 + aux_Omega_b7,r4 + aux_Omega_b7,r5
        + aux_Omega_b7,r6 + aux_Omega_b7,r7 + aux_Omega_b7,r8 + aux_Omega_b7,r9
        - GlobalBidder_PSI__i7,k2 - 18497526.8940961 GlobalBidder_hat{W}__i7,k2
        >= -18497526.8940961
 c4523: 0.0102040816326531 X__b9,r7,band_LOW_PAIRED - GlobalBidder_W_ir__b9,r7
        + 0.0102040816326531 X__b9,r7,band_HIGH_PAIRED
        + 0.0102040816326531 X__b9,r7,band_UNPAIRED <= 0
 c4524: - sv_function_helpervar_b8,r12__condX2
        + 0.159946187470694 sv_function_helpervar_b8,r12__Z2 <= 0
 c4525: aux_cap_helper__b2,r6,band_UNPAIRED__Z1
        + aux_cap_helper__b2,r6,band_UNPAIRED__Z2  = 1
 c4526: aux_cap_helper__b3,r13,band_LOW_PAIRED__Z2
        + aux_cap_helper__b3,r13,band_LOW_PAIRED__Z1  = 1
 c4527: aux_cap_helper__b7,r12,band_LOW_PAIRED__Z2
        + aux_cap_helper__b7,r12,band_LOW_PAIRED__Z1  = 1
 c4528: - X__b9,r9,band_LOW_PAIRED
        + aux_cap_helper__b9,r9,band_LOW_PAIRED__condX1
        + aux_cap_helper__b9,r9,band_LOW_PAIRED__condX2  = 0
 c4529: sv_function_helpervar_b3,r9__condX3
        - 15.9946187432626 sv_function_helpervar_b3,r9__Z3 <= 0
 c4530: - aux_cap_helper__b6,r11,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b6,r11,band_HIGH_PAIRED__Z3 <= 0
 c4531: - aux_cap_helper__b3,r2,band_LOW_PAIRED__Z1
        + aux_cap_helper__b3,r2,band_LOW_PAIRED__condX1 <= 0
 c4532: - aux_cap_helper__b0,r4,band_UNPAIRED__condX1 <= 0
 c4533: - X__b4,r0,band_UNPAIRED + aux_cap_helper__b4,r0,band_UNPAIRED__condX2
        + aux_cap_helper__b4,r0,band_UNPAIRED__condX1  = 0
 c4534: - 0.159946187432665 sv_function_helpervar_b0,r9__Z1
        + sv_function_helpervar_b0,r9__condX1 <= 0
 c4535: - aux_cap_helper__b2,r1,band_UNPAIRED__condX1 <= 0
 c4536: - aux_Omega_b5,r3 + 359.988981606135 aux_sv_b5,r3  = 0
 c4537: - aux_Omega_b5,r9 + 303.32367107554 aux_sv_b5,r9  = 0
 c4538: - aux_Omega_b6,r6 + 36.107318376489 aux_sv_b6,r6  = 0
 c4539: - X__b8,r0,band_HIGH_PAIRED
        + aux_cap_helper__b8,r0,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b8,r0,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b8,r0,band_HIGH_PAIRED__condX1  = 0
 c4540: 0.159946187453678 sv_function_helpervar_b4,r11__Z2
        - sv_function_helpervar_b4,r11__condX2 <= 0
 c4541: aux_cap_helper__b7,r10,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b7,r10,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b7,r10,band_HIGH_PAIRED__Z3  = 1
 c4542: - aux_cap_b9,r11,band_LOW_PAIRED
        + 3.735962531 aux_cap_helper__b9,r11,band_LOW_PAIRED__condX1
        + 5.2303475434 aux_cap_helper__b9,r11,band_LOW_PAIRED__condX2
        - 1.4943850124 aux_cap_helper__b9,r11,band_LOW_PAIRED__Z2  = 0
 c4543: - sv_function_helpervar_b6,r2__condX2
        + 0.159946187456351 sv_function_helpervar_b6,r2__Z2 <= 0
 c4544: - X__b7,r12,band_HIGH_PAIRED
        + aux_cap_helper__b7,r12,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b7,r12,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b7,r12,band_HIGH_PAIRED__condX2  = 0
 c4545: sv_function_helpervar_b8,r0__condX3
        - 15.9946187432626 sv_function_helpervar_b8,r0__Z3 <= 0
 c4546: aux_cap_helper__b1,r11,band_LOW_PAIRED__Z2
        + aux_cap_helper__b1,r11,band_LOW_PAIRED__Z1  = 1
 c4547: GlobalBidder_W_i_9 + 13 GlobalBidder_hat{W}__i9,k2 >= 13
 c4548: - aux_cap_helper__b4,r9,band_LOW_PAIRED__condX2
        + aux_cap_helper__b4,r9,band_LOW_PAIRED__Z2 <= 0
 c4549: aux_cap_helper__b8,r8,band_LOW_PAIRED__Z1
        + aux_cap_helper__b8,r8,band_LOW_PAIRED__Z2  = 1
 c4550: aux_cap_helper__b6,r7,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b6,r7,band_LOW_PAIRED__Z2 <= 0
 c4551: - aux_cap_b3,r12,band_HIGH_PAIRED
        - 0.6181448971 aux_cap_helper__b3,r12,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b3,r12,band_HIGH_PAIRED__Z3
        + 1.3908260184 aux_cap_helper__b3,r12,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b3,r12,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b3,r12,band_HIGH_PAIRED__condX1  = 0
 c4552: - X__b7,r13,band_LOW_PAIRED
        + aux_cap_helper__b7,r13,band_LOW_PAIRED__condX2
        + aux_cap_helper__b7,r13,band_LOW_PAIRED__condX1  = 0
 c4553: sv_function_helpervar_b2,r11__condX3
        + sv_function_helpervar_b2,r11__condX2
        + sv_function_helpervar_b2,r11__condX1 - aux_c_b2,r11  = 0
 c4554: 1.1250244065 aux_cap_helper__b0,r10,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b0,r10,band_UNPAIRED__condX1
        - aux_cap_b0,r10,band_UNPAIRED
        - 0.3214355447 aux_cap_helper__b0,r10,band_UNPAIRED__Z2  = 0
 c4555: - sv_function_helpervar_b9,r6__condX1 <= 0
 c4556: aux_cap_helper__b9,r5,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b9,r5,band_UNPAIRED__Z2 <= 0
 c4557: - aux_cap_helper__b4,r9,band_LOW_PAIRED__condX1 <= 0
 c4558: - sv_function_helpervar_b0,r12__condX2
        + 0.159946187446023 sv_function_helpervar_b0,r12__Z2 <= 0
 c4559: - X__b4,r4,band_LOW_PAIRED
        + aux_cap_helper__b4,r4,band_LOW_PAIRED__condX1
        + aux_cap_helper__b4,r4,band_LOW_PAIRED__condX2  = 0
 c4560: - aux_cap_helper__b1,r4,band_HIGH_PAIRED__condX1 <= 0
 c4561: aux_cap_b2,r3,band_LOW_PAIRED + aux_cap_b2,r3,band_UNPAIRED
        + aux_cap_b2,r3,band_HIGH_PAIRED - aux_c_b2,r3  = 0
 c4562: aux_cap_helper__b2,r1,band_UNPAIRED__Z2
        - aux_cap_helper__b2,r1,band_UNPAIRED__condX2 <= 0
 c4563: sv_function_helpervar_b2,r6__Z1 + sv_function_helpervar_b2,r6__Z2
        + sv_function_helpervar_b2,r6__Z3  = 1
 c4564: sv_function_helpervar_b6,r11__Z3 + sv_function_helpervar_b6,r11__Z2
        + sv_function_helpervar_b6,r11__Z1  = 1
 c4565: - aux_Omega_b3,r0 + 121.38382681251 aux_sv_b3,r0  = 0
 c4566: aux_cap_helper__b6,r13,band_UNPAIRED__Z2
        + aux_cap_helper__b6,r13,band_UNPAIRED__Z1  = 1
 c4567: aux_cap_helper__b6,r12,band_UNPAIRED__Z2
        + aux_cap_helper__b6,r12,band_UNPAIRED__Z1  = 1
 c4568: - aux_cap_helper__b0,r13,band_HIGH_PAIRED__condX1 <= 0
 c4569: aux_cap_helper__b9,r5,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b9,r5,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b9,r5,band_HIGH_PAIRED__Z3  = 1
 c4570: - aux_cap_helper__b0,r10,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b0,r10,band_HIGH_PAIRED__Z2 <= 0
 c4571: 1.1250244065 aux_cap_helper__b5,r9,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b5,r9,band_UNPAIRED__condX1
        - aux_cap_b5,r9,band_UNPAIRED
        - 0.3214355447 aux_cap_helper__b5,r9,band_UNPAIRED__Z2  = 0
 c4572: - aux_cap_helper__b8,r0,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b8,r0,band_HIGH_PAIRED__Z3 <= 0
 c4573: aux_cap_helper__b0,r7,band_LOW_PAIRED__condX2
        + aux_cap_helper__b0,r7,band_LOW_PAIRED__condX1
        - X__b0,r7,band_LOW_PAIRED  = 0
 c4574: - aux_cap_helper__b0,r2,band_LOW_PAIRED__Z1
        + aux_cap_helper__b0,r2,band_LOW_PAIRED__condX1 <= 0
 c4575: aux_cap_helper__b7,r12,band_LOW_PAIRED__condX1
        + aux_cap_helper__b7,r12,band_LOW_PAIRED__condX2
        - X__b7,r12,band_LOW_PAIRED  = 0
 c4576: aux_cap_helper__b8,r12,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b8,r12,band_LOW_PAIRED__Z2 <= 0
 c4577: - 15.9946187432626 sv_function_helpervar_b3,r12__Z3
        + sv_function_helpervar_b3,r12__condX3 <= 0
 c4578: 0.15994618744377 sv_function_helpervar_b4,r12__Z2
        - sv_function_helpervar_b4,r12__condX2 <= 0
 c4579: aux_cap_helper__b1,r3,band_LOW_PAIRED__Z1
        + aux_cap_helper__b1,r3,band_LOW_PAIRED__Z2  = 1
 c4580: aux_cap_helper__b3,r10,band_UNPAIRED__condX1
        - aux_cap_helper__b3,r10,band_UNPAIRED__Z1 <= 0
 c4581: 5.74126635300054 sv_function_helpervar_b3,r1__Z3
        - sv_function_helpervar_b3,r1__condX3 <= 0
 c4582: 1.1250244065 aux_cap_helper__b4,r4,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b4,r4,band_UNPAIRED__condX1
        - aux_cap_b4,r4,band_UNPAIRED
        - 0.3214355447 aux_cap_helper__b4,r4,band_UNPAIRED__Z2  = 0
 c4583: - sv_function_helpervar_b2,r4__condX3
        + 6.57835953923278 sv_function_helpervar_b2,r4__Z3 <= 0
 c4584: sv_function_helpervar_b8,r2__Z3 + sv_function_helpervar_b8,r2__Z2
        + sv_function_helpervar_b8,r2__Z1  = 1
 c4585: sv_function_helpervar_b9,r1__condX1
        + sv_function_helpervar_b9,r1__condX2
        + sv_function_helpervar_b9,r1__condX3 - aux_c_b9,r1  = 0
 c4586: - 0.3214355447 aux_cap_helper__b7,r3,band_UNPAIRED__Z2
        - aux_cap_b7,r3,band_UNPAIRED
        + 0.8035888618 aux_cap_helper__b7,r3,band_UNPAIRED__condX1
        + 1.1250244065 aux_cap_helper__b7,r3,band_UNPAIRED__condX2  = 0
 c4587: aux_cap_helper__b4,r4,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b4,r4,band_HIGH_PAIRED__condX2 <= 0
 c4588: - aux_cap_helper__b2,r10,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b2,r10,band_HIGH_PAIRED__Z3 <= 0
 c4589: - aux_cap_helper__b0,r10,band_HIGH_PAIRED__condX1 <= 0
 c4590: - X__b9,r2,band_HIGH_PAIRED
        + aux_cap_helper__b9,r2,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b9,r2,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b9,r2,band_HIGH_PAIRED__condX3  = 0
 c4591: - 3 aux_cap_helper__b9,r8,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b9,r8,band_HIGH_PAIRED__condX3 <= 0
 c4592: - aux_cap_b5,r4,band_LOW_PAIRED
        - 1.4943850124 aux_cap_helper__b5,r4,band_LOW_PAIRED__Z2
        + 5.2303475434 aux_cap_helper__b5,r4,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b5,r4,band_LOW_PAIRED__condX1  = 0
 c4593: - aux_cap_helper__b1,r7,band_HIGH_PAIRED__condX1 <= 0
 c4594: - X__b8,r0,band_UNPAIRED + aux_cap_helper__b8,r0,band_UNPAIRED__condX1
        + aux_cap_helper__b8,r0,band_UNPAIRED__condX2  = 0
 c4595: - 2 aux_cap_helper__b7,r9,band_LOW_PAIRED__Z2
        + aux_cap_helper__b7,r9,band_LOW_PAIRED__condX2 <= 0
 c4596: aux_cap_helper__b1,r4,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b1,r4,band_HIGH_PAIRED__condX2 <= 0
 c4597: - aux_cap_helper__b9,r0,band_LOW_PAIRED__Z1
        + aux_cap_helper__b9,r0,band_LOW_PAIRED__condX1 <= 0
 c4598: - sv_function_helpervar_b1,r8__condX1 <= 0
 c4599: - aux_cap_b8,r11,band_HIGH_PAIRED
        + 0.927217345653617 aux_cap_helper__b8,r11,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b8,r11,band_HIGH_PAIRED__Z2
        + 1.5453622427 aux_cap_helper__b8,r11,band_HIGH_PAIRED__condX1
        + 1.3908260184 aux_cap_helper__b8,r11,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b8,r11,band_HIGH_PAIRED__condX2  = 0
 c4600: - 2 aux_cap_helper__b9,r4,band_LOW_PAIRED__Z2
        + aux_cap_helper__b9,r4,band_LOW_PAIRED__condX2 <= 0
 c4601: aux_cap_b2,r13,band_UNPAIRED + aux_cap_b2,r13,band_HIGH_PAIRED
        - aux_c_b2,r13 + aux_cap_b2,r13,band_LOW_PAIRED  = 0
 c4602: sv_function_helpervar_b4,r7__Z1 + sv_function_helpervar_b4,r7__Z3
        + sv_function_helpervar_b4,r7__Z2  = 1
 c4603: aux_cap_helper__b8,r1,band_LOW_PAIRED__Z2
        + aux_cap_helper__b8,r1,band_LOW_PAIRED__Z1  = 1
 c4604: - aux_Omega_b2,r7 + 2015.51832942916 aux_sv_b2,r7  = 0
 c4605: aux_cap_helper__b4,r3,band_LOW_PAIRED__condX1
        + aux_cap_helper__b4,r3,band_LOW_PAIRED__condX2
        - X__b4,r3,band_LOW_PAIRED  = 0
 c4606: - aux_cap_helper__b6,r5,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b6,r5,band_HIGH_PAIRED__condX1 <= 0
 c4607: aux_cap_helper__b4,r1,band_LOW_PAIRED__Z2
        + aux_cap_helper__b4,r1,band_LOW_PAIRED__Z1  = 1
 c4608: - aux_cap_helper__b6,r2,band_UNPAIRED__Z1
        + aux_cap_helper__b6,r2,band_UNPAIRED__condX1 <= 0
 c4609: aux_cap_helper__b2,r6,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b2,r6,band_HIGH_PAIRED__Z2 <= 0
 c4610: - 0.6181448971 aux_cap_helper__b1,r10,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b1,r10,band_HIGH_PAIRED__Z3
        - aux_cap_b1,r10,band_HIGH_PAIRED
        + 1.3908260184 aux_cap_helper__b1,r10,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b1,r10,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b1,r10,band_HIGH_PAIRED__condX1  = 0
 c4611: X__b0,r7,band_UNPAIRED + X__b2,r7,band_UNPAIRED
        + X__b1,r7,band_UNPAIRED + X__b7,r7,band_UNPAIRED
        + X__b5,r7,band_UNPAIRED + X__b6,r7,band_UNPAIRED
        + X__b3,r7,band_UNPAIRED + X__b8,r7,band_UNPAIRED
        + X__b9,r7,band_UNPAIRED + X__b4,r7,band_UNPAIRED <= 2
 c4612: 2 aux_cap_helper__b6,r8,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b6,r8,band_HIGH_PAIRED__condX3 <= 0
 c4613: - aux_c_b2,r0 + sv_function_helpervar_b2,r0__condX1
        + sv_function_helpervar_b2,r0__condX2
        + sv_function_helpervar_b2,r0__condX3  = 0
 c4614: aux_cap_helper__b9,r11,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b9,r11,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b9,r11,band_HIGH_PAIRED__condX3
        - X__b9,r11,band_HIGH_PAIRED  = 0
 c4615: - 3 aux_cap_helper__b5,r0,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b5,r0,band_HIGH_PAIRED__condX3 <= 0
 c4616: - sv_function_helpervar_b1,r7__condX1 <= 0
 c4617: GlobalBidder_W_i_9 + 14 GlobalBidder_hat{W}__i9,k1 <= 27
 c4618: aux_cap_helper__b8,r11,band_UNPAIRED__condX1
        - aux_cap_helper__b8,r11,band_UNPAIRED__Z1 <= 0
 c4619: - aux_cap_helper__b3,r12,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b3,r12,band_HIGH_PAIRED__condX1 <= 0
 c4620: - aux_cap_helper__b8,r11,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b8,r11,band_HIGH_PAIRED__condX1 <= 0
 c4621: aux_cap_helper__b0,r8,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b0,r8,band_HIGH_PAIRED__Z3 <= 0
 c4622: aux_cap_helper__b1,r7,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b1,r7,band_HIGH_PAIRED__condX2 <= 0
 c4623: aux_cap_helper__b3,r5,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b3,r5,band_HIGH_PAIRED__Z3 <= 0
 c4624: aux_cap_helper__b2,r9,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b2,r9,band_HIGH_PAIRED__Z2 <= 0
 c4625: X__b9,r8,band_HIGH_PAIRED + X__b2,r8,band_HIGH_PAIRED
        + X__b1,r8,band_HIGH_PAIRED + X__b7,r8,band_HIGH_PAIRED
        + X__b0,r8,band_HIGH_PAIRED + X__b8,r8,band_HIGH_PAIRED
        + X__b6,r8,band_HIGH_PAIRED + X__b3,r8,band_HIGH_PAIRED
        + X__b4,r8,band_HIGH_PAIRED + X__b5,r8,band_HIGH_PAIRED <= 3
 c4626: aux_cap_helper__b0,r6,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b0,r6,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b0,r6,band_HIGH_PAIRED__Z1  = 1
 c4627: - 1.4943850124 aux_cap_helper__b1,r0,band_LOW_PAIRED__Z2
        + 3.735962531 aux_cap_helper__b1,r0,band_LOW_PAIRED__condX1
        + 5.2303475434 aux_cap_helper__b1,r0,band_LOW_PAIRED__condX2
        - aux_cap_b1,r0,band_LOW_PAIRED  = 0
 c4628: sv_function_helpervar_b6,r7__condX3
        - 15.9946187432626 sv_function_helpervar_b6,r7__Z3 <= 0
 c4629: - aux_cap_helper__b7,r1,band_LOW_PAIRED__condX2
        + aux_cap_helper__b7,r1,band_LOW_PAIRED__Z2 <= 0
 c4630: sv_function_helpervar_b1,r4__condX1
        - 0.159946187439907 sv_function_helpervar_b1,r4__Z1 <= 0
 c4631: aux_cap_helper__b2,r10,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b2,r10,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b2,r10,band_HIGH_PAIRED__Z3  = 1
 c4632: - aux_c_b0,r9 + sv_function_helpervar_b0,r9__condX1
        + sv_function_helpervar_b0,r9__condX3
        + sv_function_helpervar_b0,r9__condX2  = 0
 c4633: aux_cap_helper__b2,r6,band_UNPAIRED__condX1
        - aux_cap_helper__b2,r6,band_UNPAIRED__Z1 <= 0
 c4634: aux_cap_helper__b1,r5,band_UNPAIRED__Z1
        + aux_cap_helper__b1,r5,band_UNPAIRED__Z2  = 1
 c4635: - 2 aux_cap_helper__b5,r6,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b5,r6,band_HIGH_PAIRED__condX2 <= 0
 c4636: aux_cap_helper__b7,r10,band_UNPAIRED__Z2
        + aux_cap_helper__b7,r10,band_UNPAIRED__Z1  = 1
 c4637: - X__b8,r0,band_LOW_PAIRED
        + aux_cap_helper__b8,r0,band_LOW_PAIRED__condX2
        + aux_cap_helper__b8,r0,band_LOW_PAIRED__condX1  = 0
 c4638: - aux_cap_helper__b0,r13,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b0,r13,band_HIGH_PAIRED__Z2 <= 0
 c4639: - aux_cap_helper__b8,r3,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b8,r3,band_HIGH_PAIRED__Z3 <= 0
 c4640: - aux_cap_helper__b1,r2,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b1,r2,band_HIGH_PAIRED__Z3 <= 0
 c4641: - 1.4943850124 aux_cap_helper__b5,r8,band_LOW_PAIRED__Z2
        - aux_cap_b5,r8,band_LOW_PAIRED
        + 3.735962531 aux_cap_helper__b5,r8,band_LOW_PAIRED__condX1
        + 5.2303475434 aux_cap_helper__b5,r8,band_LOW_PAIRED__condX2  = 0
 c4642: - aux_cap_helper__b1,r9,band_LOW_PAIRED__condX1 <= 0
 c4643: - aux_cap_helper__b8,r0,band_LOW_PAIRED__condX1 <= 0
 c4644: - 3 aux_cap_helper__b3,r8,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b3,r8,band_HIGH_PAIRED__condX3 <= 0
 c4645: - sv_function_helpervar_b6,r10__condX1 <= 0
 c4646: aux_cap_helper__b8,r8,band_UNPAIRED__condX1
        - aux_cap_helper__b8,r8,band_UNPAIRED__Z1 <= 0
 c4647: aux_cap_helper__b0,r12,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b0,r12,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b0,r12,band_HIGH_PAIRED__Z2  = 1
 c4648: - aux_cap_helper__b2,r10,band_UNPAIRED__Z1
        + aux_cap_helper__b2,r10,band_UNPAIRED__condX1 <= 0
 c4649: - aux_cap_b8,r1,band_HIGH_PAIRED
        + 1.5453622427 aux_cap_helper__b8,r1,band_HIGH_PAIRED__condX1
        + 2.1635071398 aux_cap_helper__b8,r1,band_HIGH_PAIRED__condX2
        + 1.3908260184 aux_cap_helper__b8,r1,band_HIGH_PAIRED__condX3
        - 0.6181448971 aux_cap_helper__b8,r1,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b8,r1,band_HIGH_PAIRED__Z3  = 0
 c4650: aux_cap_b2,r8,band_LOW_PAIRED + aux_cap_b2,r8,band_HIGH_PAIRED
        - aux_c_b2,r8 + aux_cap_b2,r8,band_UNPAIRED  = 0
 c4651: 0.0102040816326531 X__b7,r5,band_UNPAIRED
        + 0.0102040816326531 X__b7,r5,band_HIGH_PAIRED
        - GlobalBidder_W_ir__b7,r5
        + 0.0102040816326531 X__b7,r5,band_LOW_PAIRED <= 0
 c4652: - X__b7,r4,band_LOW_PAIRED
        + aux_cap_helper__b7,r4,band_LOW_PAIRED__condX2
        + aux_cap_helper__b7,r4,band_LOW_PAIRED__condX1  = 0
 c4653: aux_cap_b6,r0,band_UNPAIRED + aux_cap_b6,r0,band_HIGH_PAIRED
        - aux_c_b6,r0 + aux_cap_b6,r0,band_LOW_PAIRED  = 0
 c4654: aux_cap_helper__b5,r6,band_UNPAIRED__condX1
        - aux_cap_helper__b5,r6,band_UNPAIRED__Z1 <= 0
 c4655: - 1.4943850124 aux_cap_helper__b9,r1,band_LOW_PAIRED__Z2
        - aux_cap_b9,r1,band_LOW_PAIRED
        + 5.2303475434 aux_cap_helper__b9,r1,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b9,r1,band_LOW_PAIRED__condX1  = 0
 c4656: aux_cap_helper__b2,r8,band_UNPAIRED__Z2
        - aux_cap_helper__b2,r8,band_UNPAIRED__condX2 <= 0
 c4657: aux_cap_helper__b5,r12,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b5,r12,band_LOW_PAIRED__Z2 <= 0
 c4658: - aux_cap_b4,r6,band_LOW_PAIRED
        - 1.4943850124 aux_cap_helper__b4,r6,band_LOW_PAIRED__Z2
        + 5.2303475434 aux_cap_helper__b4,r6,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b4,r6,band_LOW_PAIRED__condX1  = 0
 c4659: - 2 aux_cap_helper__b1,r8,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b1,r8,band_HIGH_PAIRED__condX2 <= 0
 c4660: - aux_cap_helper__b2,r8,band_LOW_PAIRED__condX1 <= 0
 c4661: aux_cap_b3,r6,band_LOW_PAIRED + aux_cap_b3,r6,band_HIGH_PAIRED
        + aux_cap_b3,r6,band_UNPAIRED - aux_c_b3,r6  = 0
 c4662: - sv_function_helpervar_b3,r0__condX1 <= 0
 c4663: aux_cap_helper__b3,r3,band_LOW_PAIRED__Z1
        + aux_cap_helper__b3,r3,band_LOW_PAIRED__Z2  = 1
 c4664: 265.665791727752 sv_function_helpervar_b9,r8__Z2
        + 452.745530991097 sv_function_helpervar_b9,r8__Z3
        + 35.3640540951 sv_function_helpervar_b9,r8__condX3
        + 58.1247584217 sv_function_helpervar_b9,r8__condX2
        + 1719.0945881521 sv_function_helpervar_b9,r8__condX1 - aux_sv_b9,r8
         = 0
 c4665: aux_cap_b5,r2,band_LOW_PAIRED + aux_cap_b5,r2,band_HIGH_PAIRED
        + aux_cap_b5,r2,band_UNPAIRED - aux_c_b5,r2  = 0
 c4666: - sv_function_helpervar_b0,r12__condX3
        + 5.99016947043189 sv_function_helpervar_b0,r12__Z3 <= 0
 c4667: - aux_cap_helper__b9,r11,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b9,r11,band_HIGH_PAIRED__Z3 <= 0
 c4668: - 3 aux_cap_helper__b4,r13,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b4,r13,band_HIGH_PAIRED__condX3 <= 0
 c4669: - aux_cap_helper__b4,r4,band_HIGH_PAIRED__condX1 <= 0
 c4670: - 0.6181448971 aux_cap_helper__b7,r1,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b7,r1,band_HIGH_PAIRED__Z3
        - aux_cap_b7,r1,band_HIGH_PAIRED
        + 2.1635071398 aux_cap_helper__b7,r1,band_HIGH_PAIRED__condX2
        + 1.3908260184 aux_cap_helper__b7,r1,band_HIGH_PAIRED__condX3
        + 1.5453622427 aux_cap_helper__b7,r1,band_HIGH_PAIRED__condX1  = 0
 c4671: sv_function_helpervar_b1,r7__condX3
        - 15.9946187432626 sv_function_helpervar_b1,r7__Z3 <= 0
 c4672: 0.0102040816326531 X__b9,r1,band_UNPAIRED
        + 0.0102040816326531 X__b9,r1,band_LOW_PAIRED
        + 0.0102040816326531 X__b9,r1,band_HIGH_PAIRED
        - GlobalBidder_W_ir__b9,r1 <= 0
 c4673: - sv_function_helpervar_b4,r9__condX3
        + 5.68428432618719 sv_function_helpervar_b4,r9__Z3 <= 0
 c4674: aux_cap_helper__b9,r7,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b9,r7,band_HIGH_PAIRED__Z2 <= 0
 c4675: - 0.159946187489304 sv_function_helpervar_b9,r1__Z1
        + sv_function_helpervar_b9,r1__condX1 <= 0
 c4676: sv_function_helpervar_b8,r6__Z3 + sv_function_helpervar_b8,r6__Z2
        + sv_function_helpervar_b8,r6__Z1  = 1
 c4677: aux_cap_helper__b2,r7,band_UNPAIRED__Z1
        + aux_cap_helper__b2,r7,band_UNPAIRED__Z2  = 1
 c4678: - aux_cap_helper__b2,r5,band_HIGH_PAIRED__condX1 <= 0
 c4679: sv_function_helpervar_b5,r4__condX1
        - 0.159946187436471 sv_function_helpervar_b5,r4__Z1 <= 0
 c4680: aux_cap_helper__b8,r9,band_UNPAIRED__condX2
        + aux_cap_helper__b8,r9,band_UNPAIRED__condX1 - X__b8,r9,band_UNPAIRED
         = 0
 c4681: aux_cap_helper__b4,r4,band_LOW_PAIRED__condX1
        - aux_cap_helper__b4,r4,band_LOW_PAIRED__Z1 <= 0
 c4682: aux_cap_helper__b5,r13,band_UNPAIRED__condX1
        - aux_cap_helper__b5,r13,band_UNPAIRED__Z1 <= 0
 c4683: - aux_Omega_b3,r5 + 630.970474387529 aux_sv_b3,r5  = 0
 c4684: - aux_c_b0,r13 + aux_cap_b0,r13,band_UNPAIRED
        + aux_cap_b0,r13,band_LOW_PAIRED + aux_cap_b0,r13,band_HIGH_PAIRED  = 0
 c4685: - X__b0,r0,band_LOW_PAIRED
        + aux_cap_helper__b0,r0,band_LOW_PAIRED__condX2
        + aux_cap_helper__b0,r0,band_LOW_PAIRED__condX1  = 0
 c4686: 3.735962531 aux_cap_helper__b2,r2,band_LOW_PAIRED__condX1
        + 5.2303475434 aux_cap_helper__b2,r2,band_LOW_PAIRED__condX2
        - 1.4943850124 aux_cap_helper__b2,r2,band_LOW_PAIRED__Z2
        - aux_cap_b2,r2,band_LOW_PAIRED  = 0
 c4687: GlobalBidder_W_i_7 + 14 GlobalBidder_hat{W}__i7,k1 <= 27
 c4688: sv_function_helpervar_b8,r2__condX1
        - 0.159946187437224 sv_function_helpervar_b8,r2__Z1 <= 0
 c4689: - sv_function_helpervar_b7,r5__condX2
        + 0.159946187454911 sv_function_helpervar_b7,r5__Z2 <= 0
 c4690: 197.108191451093 sv_function_helpervar_b4,r3__Z2
        + 422.66241255657 sv_function_helpervar_b4,r3__Z3
        + 1289.3272635384 sv_function_helpervar_b4,r3__condX1
        + 21.3275732502 sv_function_helpervar_b4,r3__condX3
        + 56.9865956563 sv_function_helpervar_b4,r3__condX2 - aux_sv_b4,r3  = 0
 c4691: aux_cap_helper__b1,r3,band_UNPAIRED__condX1
        - aux_cap_helper__b1,r3,band_UNPAIRED__Z1 <= 0
 c4692: - aux_cap_helper__b9,r1,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b9,r1,band_HIGH_PAIRED__Z3 <= 0
 c4693: sv_function_helpervar_b6,r12__condX2
        - 5.83628230872818 sv_function_helpervar_b6,r12__Z2 <= 0
 c4694: aux_cap_helper__b6,r8,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b6,r8,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b6,r8,band_HIGH_PAIRED__Z1  = 1
 c4695: - aux_cap_helper__b0,r7,band_UNPAIRED__Z1
        + aux_cap_helper__b0,r7,band_UNPAIRED__condX1 <= 0
 c4696: aux_cap_helper__b3,r2,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b3,r2,band_HIGH_PAIRED__Z1 <= 0
 c4697: - X__b3,r0,band_HIGH_PAIRED
        + aux_cap_helper__b3,r0,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b3,r0,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b3,r0,band_HIGH_PAIRED__condX1  = 0
 c4698: aux_cap_helper__b5,r9,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b5,r9,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b5,r9,band_HIGH_PAIRED__Z1  = 1
 c4699: - 1.4943850124 aux_cap_helper__b2,r11,band_LOW_PAIRED__Z2
        + 5.2303475434 aux_cap_helper__b2,r11,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b2,r11,band_LOW_PAIRED__condX1
        - aux_cap_b2,r11,band_LOW_PAIRED  = 0
 c4700: aux_cap_helper__b7,r7,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b7,r7,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b7,r7,band_HIGH_PAIRED__Z1  = 1
 c4701: - sv_function_helpervar_b4,r2__condX3
        + 5.59831642723657 sv_function_helpervar_b4,r2__Z3 <= 0
 c4702: aux_cap_helper__b0,r11,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b0,r11,band_LOW_PAIRED__Z2 <= 0
 c4703: aux_cap_helper__b8,r6,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b8,r6,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b8,r6,band_HIGH_PAIRED__Z2  = 1
 c4704: 1101.82413595027 aux_sv_b2,r12 - aux_Omega_b2,r12  = 0
 c4705: - aux_cap_helper__b3,r1,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b3,r1,band_HIGH_PAIRED__Z3 <= 0
 c4706: - aux_cap_helper__b1,r9,band_LOW_PAIRED__condX2
        + aux_cap_helper__b1,r9,band_LOW_PAIRED__Z2 <= 0
 c4707: - aux_cap_helper__b2,r5,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b2,r5,band_HIGH_PAIRED__Z2 <= 0
 c4708: aux_cap_b4,r4,band_UNPAIRED + aux_cap_b4,r4,band_LOW_PAIRED
        - aux_c_b4,r4 + aux_cap_b4,r4,band_HIGH_PAIRED  = 0
 c4709: - aux_cap_helper__b5,r11,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b5,r11,band_HIGH_PAIRED__condX1 <= 0
 c4710: - 3 aux_cap_helper__b5,r7,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b5,r7,band_HIGH_PAIRED__condX3 <= 0
 c4711: - aux_cap_helper__b8,r0,band_LOW_PAIRED__condX2
        + aux_cap_helper__b8,r0,band_LOW_PAIRED__Z2 <= 0
 c4712: 42.0132169578624 sv_function_helpervar_b1,r13__Z3
        + 19.1575636594088 sv_function_helpervar_b1,r13__Z2 - aux_sv_b1,r13
        + 2.0237940116 sv_function_helpervar_b1,r13__condX3
        + 5.7885380946 sv_function_helpervar_b1,r13__condX2
        + 125.5635946858 sv_function_helpervar_b1,r13__condX1  = 0
 c4713: aux_cap_helper__b7,r8,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b7,r8,band_HIGH_PAIRED__Z2 <= 0
 c4714: aux_cap_helper__b9,r5,band_UNPAIRED__Z2
        + aux_cap_helper__b9,r5,band_UNPAIRED__Z1  = 1
 c4715: - aux_cap_helper__b0,r6,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b0,r6,band_HIGH_PAIRED__Z2 <= 0
 c4716: - 2 aux_cap_helper__b6,r11,band_LOW_PAIRED__Z2
        + aux_cap_helper__b6,r11,band_LOW_PAIRED__condX2 <= 0
 c4717: aux_cap_helper__b9,r2,band_LOW_PAIRED__condX2
        + aux_cap_helper__b9,r2,band_LOW_PAIRED__condX1
        - X__b9,r2,band_LOW_PAIRED  = 0
 c4718: aux_cap_helper__b1,r4,band_LOW_PAIRED__condX1
        - aux_cap_helper__b1,r4,band_LOW_PAIRED__Z1 <= 0
 c4719: aux_cap_helper__b2,r6,band_LOW_PAIRED__condX1
        - aux_cap_helper__b2,r6,band_LOW_PAIRED__Z1 <= 0
 c4720: - aux_cap_b2,r7,band_UNPAIRED
        - 0.3214355447 aux_cap_helper__b2,r7,band_UNPAIRED__Z2
        + 0.8035888618 aux_cap_helper__b2,r7,band_UNPAIRED__condX1
        + 1.1250244065 aux_cap_helper__b2,r7,band_UNPAIRED__condX2  = 0
 c4721: sv_function_helpervar_b2,r12__Z3 + sv_function_helpervar_b2,r12__Z2
        + sv_function_helpervar_b2,r12__Z1  = 1
 c4722: aux_cap_helper__b5,r6,band_LOW_PAIRED__condX1
        + aux_cap_helper__b5,r6,band_LOW_PAIRED__condX2
        - X__b5,r6,band_LOW_PAIRED  = 0
 c4723: - 2 aux_cap_helper__b7,r6,band_LOW_PAIRED__Z2
        + aux_cap_helper__b7,r6,band_LOW_PAIRED__condX2 <= 0
 c4724: aux_cap_helper__b3,r7,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b3,r7,band_HIGH_PAIRED__Z2 <= 0
 c4725: - 2 aux_cap_helper__b6,r4,band_LOW_PAIRED__Z2
        + aux_cap_helper__b6,r4,band_LOW_PAIRED__condX2 <= 0
 c4726: - aux_cap_helper__b2,r11,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b2,r11,band_HIGH_PAIRED__condX1 <= 0
 c4727: - sv_function_helpervar_b1,r7__condX3
        + 6.61140040786873 sv_function_helpervar_b1,r7__Z3 <= 0
 c4728: - aux_Omega_b2,r6 + 65.909117723829 aux_sv_b2,r6  = 0
 c4729: - aux_cap_b4,r1,band_UNPAIRED
        + 1.1250244065 aux_cap_helper__b4,r1,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b4,r1,band_UNPAIRED__condX1
        - 0.3214355447 aux_cap_helper__b4,r1,band_UNPAIRED__Z2  = 0
 c4730: aux_cap_helper__b3,r7,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b3,r7,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b3,r7,band_HIGH_PAIRED__Z2  = 1
 c4731: - 0.3214355447 aux_cap_helper__b8,r10,band_UNPAIRED__Z2
        + 1.1250244065 aux_cap_helper__b8,r10,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b8,r10,band_UNPAIRED__condX1
        - aux_cap_b8,r10,band_UNPAIRED  = 0
 c4732: aux_cap_helper__b1,r11,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b1,r11,band_UNPAIRED__Z2 <= 0
 c4733: - aux_c_b3,r4 + sv_function_helpervar_b3,r4__condX1
        + sv_function_helpervar_b3,r4__condX3
        + sv_function_helpervar_b3,r4__condX2  = 0
 c4734: - aux_cap_helper__b1,r9,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b1,r9,band_HIGH_PAIRED__Z3 <= 0
 c4735: aux_cap_helper__b1,r11,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b1,r11,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b1,r11,band_HIGH_PAIRED__Z2  = 1
 c4736: - aux_cap_helper__b9,r6,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b9,r6,band_HIGH_PAIRED__Z2 <= 0
 c4737: - 2 aux_cap_helper__b9,r10,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b9,r10,band_HIGH_PAIRED__condX2 <= 0
 c4738: - aux_cap_helper__b3,r4,band_UNPAIRED__condX2
        + aux_cap_helper__b3,r4,band_UNPAIRED__Z2 <= 0
 c4739: - aux_Omega_b6,r5 + 364.57969216938 aux_sv_b6,r5  = 0
 c4740: - aux_cap_helper__b6,r13,band_UNPAIRED__Z1
        + aux_cap_helper__b6,r13,band_UNPAIRED__condX1 <= 0
 c4741: - aux_cap_helper__b9,r6,band_HIGH_PAIRED__condX1 <= 0
 c4742: - aux_cap_helper__b2,r8,band_UNPAIRED__condX1 <= 0
 c4743: - sv_function_helpervar_b7,r13__condX3
        + 6.8709716173417 sv_function_helpervar_b7,r13__Z3 <= 0
 c4744: - X__b5,r7,band_HIGH_PAIRED
        + aux_cap_helper__b5,r7,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b5,r7,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b5,r7,band_HIGH_PAIRED__condX1  = 0
 c4745: GlobalBidder_PSI__i7,k1 - 18497526.8940961 GlobalBidder_hat{W}__i7,k1
        <= 0
 c4746: aux_cap_helper__b7,r11,band_UNPAIRED__Z1
        + aux_cap_helper__b7,r11,band_UNPAIRED__Z2  = 1
 c4747: aux_cap_helper__b4,r3,band_LOW_PAIRED__Z2
        + aux_cap_helper__b4,r3,band_LOW_PAIRED__Z1  = 1
 c4748: 2 aux_cap_helper__b7,r2,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b7,r2,band_HIGH_PAIRED__condX3 <= 0
 c4749: aux_cap_helper__b8,r11,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b8,r11,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b8,r11,band_HIGH_PAIRED__Z1  = 1
 c4750: - aux_cap_helper__b3,r4,band_UNPAIRED__condX1 <= 0
 c4751: aux_cap_helper__b0,r12,band_LOW_PAIRED__condX1
        + aux_cap_helper__b0,r12,band_LOW_PAIRED__condX2
        - X__b0,r12,band_LOW_PAIRED  = 0
 c4752: sv_function_helpervar_b5,r3__condX3
        + sv_function_helpervar_b5,r3__condX1
        + sv_function_helpervar_b5,r3__condX2 - aux_c_b5,r3  = 0
 c4753: - X__b2,r5,band_UNPAIRED + aux_cap_helper__b2,r5,band_UNPAIRED__condX2
        + aux_cap_helper__b2,r5,band_UNPAIRED__condX1  = 0
 c4754: - aux_Omega_b7,r8 + 224.940116895219 aux_sv_b7,r8  = 0
 c4755: sv_function_helpervar_b2,r4__condX1
        - 0.159946187439258 sv_function_helpervar_b2,r4__Z1 <= 0
 c4756: - 2 aux_cap_helper__b0,r0,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b0,r0,band_HIGH_PAIRED__condX2 <= 0
 c4757: 5.2303475434 aux_cap_helper__b3,r4,band_LOW_PAIRED__condX2
        + 3.735962531 aux_cap_helper__b3,r4,band_LOW_PAIRED__condX1
        - aux_cap_b3,r4,band_LOW_PAIRED
        - 1.4943850124 aux_cap_helper__b3,r4,band_LOW_PAIRED__Z2  = 0
 c4758: - 2 aux_cap_helper__b6,r12,band_UNPAIRED__Z2
        + aux_cap_helper__b6,r12,band_UNPAIRED__condX2 <= 0
 c4759: - 2 aux_cap_helper__b9,r0,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b9,r0,band_HIGH_PAIRED__condX2 <= 0
 c4760: - aux_cap_helper__b4,r0,band_UNPAIRED__condX2
        + aux_cap_helper__b4,r0,band_UNPAIRED__Z2 <= 0
 c4761: sv_function_helpervar_b4,r7__condX2
        + sv_function_helpervar_b4,r7__condX3
        + sv_function_helpervar_b4,r7__condX1 - aux_c_b4,r7  = 0
 c4762: - aux_Omega_b6,r7 + 2603.74046466849 aux_sv_b6,r7  = 0
 c4763: aux_cap_helper__b2,r8,band_LOW_PAIRED__Z2
        - aux_cap_helper__b2,r8,band_LOW_PAIRED__condX2 <= 0
 c4764: - aux_cap_b6,r1,band_HIGH_PAIRED
        - 0.6181448971 aux_cap_helper__b6,r1,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b6,r1,band_HIGH_PAIRED__Z3
        + 1.3908260184 aux_cap_helper__b6,r1,band_HIGH_PAIRED__condX3
        + 1.5453622427 aux_cap_helper__b6,r1,band_HIGH_PAIRED__condX1
        + 2.1635071398 aux_cap_helper__b6,r1,band_HIGH_PAIRED__condX2  = 0
 c4765: - X__b8,r10,band_HIGH_PAIRED
        + aux_cap_helper__b8,r10,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b8,r10,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b8,r10,band_HIGH_PAIRED__condX1  = 0
 c4766: - X__b5,r4,band_UNPAIRED + aux_cap_helper__b5,r4,band_UNPAIRED__condX2
        + aux_cap_helper__b5,r4,band_UNPAIRED__condX1  = 0
 c4767: - aux_cap_helper__b1,r11,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b1,r11,band_HIGH_PAIRED__Z2 <= 0
 c4768: X__b4,r3,band_HIGH_PAIRED + X__b0,r3,band_HIGH_PAIRED
        + X__b8,r3,band_HIGH_PAIRED + X__b1,r3,band_HIGH_PAIRED
        + X__b2,r3,band_HIGH_PAIRED + X__b9,r3,band_HIGH_PAIRED
        + X__b3,r3,band_HIGH_PAIRED + X__b7,r3,band_HIGH_PAIRED
        + X__b6,r3,band_HIGH_PAIRED + X__b5,r3,band_HIGH_PAIRED <= 3
 c4769: - aux_cap_helper__b0,r6,band_HIGH_PAIRED__condX1 <= 0
 c4770: - aux_cap_b3,r4,band_UNPAIRED
        + 0.8035888618 aux_cap_helper__b3,r4,band_UNPAIRED__condX1
        + 1.1250244065 aux_cap_helper__b3,r4,band_UNPAIRED__condX2
        - 0.3214355447 aux_cap_helper__b3,r4,band_UNPAIRED__Z2  = 0
 c4771: - aux_cap_helper__b1,r11,band_HIGH_PAIRED__condX1 <= 0
 c4772: - aux_cap_helper__b8,r4,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b8,r4,band_HIGH_PAIRED__condX1 <= 0
 c4773: 22.0229849629 sv_function_helpervar_b3,r10__condX3
        + 1379.5885609006 sv_function_helpervar_b3,r10__condX1
        + 64.6482963806 sv_function_helpervar_b3,r10__condX2
        + 465.009754463033 sv_function_helpervar_b3,r10__Z3
        + 210.319682147003 sv_function_helpervar_b3,r10__Z2 - aux_sv_b3,r10
         = 0
 c4774: - 2 aux_cap_helper__b3,r0,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b3,r0,band_HIGH_PAIRED__condX2 <= 0
 c4775: - 2 aux_cap_helper__b1,r10,band_LOW_PAIRED__Z2
        + aux_cap_helper__b1,r10,band_LOW_PAIRED__condX2 <= 0
 c4776: - aux_cap_helper__b9,r4,band_UNPAIRED__Z1
        + aux_cap_helper__b9,r4,band_UNPAIRED__condX1 <= 0
 c4777: aux_cap_helper__b2,r4,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b2,r4,band_HIGH_PAIRED__Z1 <= 0
 c4778: 0.0102040816326531 X__b8,r3,band_HIGH_PAIRED
        + 0.0102040816326531 X__b8,r3,band_UNPAIRED
        + 0.0102040816326531 X__b8,r3,band_LOW_PAIRED
        - GlobalBidder_W_ir__b8,r3 <= 0
 c4779: aux_cap_helper__b6,r10,band_LOW_PAIRED__condX1
        + aux_cap_helper__b6,r10,band_LOW_PAIRED__condX2
        - X__b6,r10,band_LOW_PAIRED  = 0
 c4780: - X__b8,r3,band_UNPAIRED + aux_cap_helper__b8,r3,band_UNPAIRED__condX2
        + aux_cap_helper__b8,r3,band_UNPAIRED__condX1  = 0
 c4781: - sv_function_helpervar_b8,r0__condX1 <= 0
 c4782: aux_cap_helper__b7,r13,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b7,r13,band_HIGH_PAIRED__Z1 <= 0
 c4783: aux_cap_helper__b1,r7,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b1,r7,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b1,r7,band_HIGH_PAIRED__Z1  = 1
 c4784: - 6.01974020941575 sv_function_helpervar_b3,r2__Z2
        + sv_function_helpervar_b3,r2__condX2 <= 0
 c4785: - sv_function_helpervar_b1,r2__condX3
        + 5.77334678997269 sv_function_helpervar_b1,r2__Z3 <= 0
 c4786: - sv_function_helpervar_b6,r9__condX2
        + 0.159946187505012 sv_function_helpervar_b6,r9__Z2 <= 0
 c4787: - aux_cap_b1,r12,band_LOW_PAIRED
        - 1.4943850124 aux_cap_helper__b1,r12,band_LOW_PAIRED__Z2
        + 3.735962531 aux_cap_helper__b1,r12,band_LOW_PAIRED__condX1
        + 5.2303475434 aux_cap_helper__b1,r12,band_LOW_PAIRED__condX2  = 0
 c4788: - aux_cap_helper__b9,r2,band_LOW_PAIRED__condX1 <= 0
 c4789: 2 aux_cap_helper__b7,r9,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b7,r9,band_HIGH_PAIRED__condX3 <= 0
 c4790: 2 aux_cap_helper__b5,r10,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b5,r10,band_HIGH_PAIRED__condX3 <= 0
 c4791: X__b9,r7,band_LOW_PAIRED - GlobalBidder_W_ir__b9,r7
        + X__b9,r7,band_HIGH_PAIRED + X__b9,r7,band_UNPAIRED >= 0
 c4792: aux_cap_helper__b1,r2,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b1,r2,band_UNPAIRED__Z2 <= 0
 c4793: sv_function_helpervar_b1,r5__Z2 + sv_function_helpervar_b1,r5__Z1
        + sv_function_helpervar_b1,r5__Z3  = 1
 c4794: sv_function_helpervar_b7,r10__Z3 + sv_function_helpervar_b7,r10__Z2
        + sv_function_helpervar_b7,r10__Z1  = 1
 c4795: aux_cap_helper__b7,r10,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b7,r10,band_LOW_PAIRED__Z2 <= 0
 c4796: aux_cap_helper__b9,r1,band_UNPAIRED__Z2
        - aux_cap_helper__b9,r1,band_UNPAIRED__condX2 <= 0
 c4797: - aux_cap_helper__b7,r2,band_UNPAIRED__condX1 <= 0
 c4798: - aux_cap_helper__b3,r6,band_HIGH_PAIRED__condX1 <= 0
 c4799: - sv_function_helpervar_b5,r2__condX1 <= 0
 c4800: - aux_cap_helper__b0,r7,band_LOW_PAIRED__condX1 <= 0
 c4801: aux_cap_helper__b5,r7,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b5,r7,band_UNPAIRED__Z2 <= 0
 c4802: - sv_function_helpervar_b1,r7__condX2
        + 0.159946187439929 sv_function_helpervar_b1,r7__Z2 <= 0
 c4803: 61.8062779691 sv_function_helpervar_b3,r7__condX2
        + 1379.5885617846 sv_function_helpervar_b3,r7__condX1
        + 22.6268381053 sv_function_helpervar_b3,r7__condX3
        + 210.774252163163 sv_function_helpervar_b3,r7__Z2
        + 455.351353673813 sv_function_helpervar_b3,r7__Z3 - aux_sv_b3,r7  = 0
 c4804: aux_cap_helper__b2,r12,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b2,r12,band_HIGH_PAIRED__condX2 <= 0
 c4805: aux_cap_helper__b7,r1,band_LOW_PAIRED__Z2
        + aux_cap_helper__b7,r1,band_LOW_PAIRED__Z1  = 1
 c4806: aux_cap_helper__b4,r9,band_UNPAIRED__Z1
        + aux_cap_helper__b4,r9,band_UNPAIRED__Z2  = 1
 c4807: aux_cap_helper__b2,r8,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b2,r8,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b2,r8,band_HIGH_PAIRED__Z2  = 1
 c4808: sv_function_helpervar_b6,r6__condX2
        + sv_function_helpervar_b6,r6__condX1
        + sv_function_helpervar_b6,r6__condX3 - aux_c_b6,r6  = 0
 c4809: aux_cap_helper__b4,r3,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b4,r3,band_HIGH_PAIRED__Z1 <= 0
 c4810: sv_function_helpervar_b5,r0__condX1
        - 0.159946187449403 sv_function_helpervar_b5,r0__Z1 <= 0
 c4811: aux_cap_helper__b7,r6,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b7,r6,band_HIGH_PAIRED__Z3 <= 0
 c4812: aux_cap_helper__b2,r10,band_UNPAIRED__Z1
        + aux_cap_helper__b2,r10,band_UNPAIRED__Z2  = 1
 c4813: - 6.35577727875249 sv_function_helpervar_b3,r6__Z2
        + sv_function_helpervar_b3,r6__condX2 <= 0
 c4814: - 2 aux_cap_helper__b2,r5,band_UNPAIRED__Z2
        + aux_cap_helper__b2,r5,band_UNPAIRED__condX2 <= 0
 c4815: - aux_cap_helper__b2,r12,band_HIGH_PAIRED__condX1 <= 0
 c4816: - aux_cap_b6,r12,band_HIGH_PAIRED
        + 2.1635071398 aux_cap_helper__b6,r12,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b6,r12,band_HIGH_PAIRED__condX1
        + 1.3908260184 aux_cap_helper__b6,r12,band_HIGH_PAIRED__condX3
        + 0.927217345653617 aux_cap_helper__b6,r12,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b6,r12,band_HIGH_PAIRED__Z2  = 0
 c4817: aux_cap_helper__b6,r11,band_UNPAIRED__condX2
        + aux_cap_helper__b6,r11,band_UNPAIRED__condX1
        - X__b6,r11,band_UNPAIRED  = 0
 c4818: aux_cap_helper__b1,r3,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b1,r3,band_HIGH_PAIRED__Z1 <= 0
 c4819: aux_cap_helper__b8,r4,band_UNPAIRED__Z2
        + aux_cap_helper__b8,r4,band_UNPAIRED__Z1  = 1
 c4820: - X__b3,r8,band_LOW_PAIRED
        + aux_cap_helper__b3,r8,band_LOW_PAIRED__condX1
        + aux_cap_helper__b3,r8,band_LOW_PAIRED__condX2  = 0
 c4821: - 15.9946187432626 sv_function_helpervar_b0,r1__Z3
        + sv_function_helpervar_b0,r1__condX3 <= 0
 c4822: - aux_c_b8,r5 + sv_function_helpervar_b8,r5__condX1
        + sv_function_helpervar_b8,r5__condX2
        + sv_function_helpervar_b8,r5__condX3  = 0
 c4823: - sv_function_helpervar_b0,r2__condX1 <= 0
 c4824: aux_cap_helper__b3,r8,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b3,r8,band_UNPAIRED__Z2 <= 0
 c4825: aux_cap_helper__b9,r2,band_LOW_PAIRED__Z2
        - aux_cap_helper__b9,r2,band_LOW_PAIRED__condX2 <= 0
 c4826: 1.5453622427 aux_cap_helper__b2,r7,band_HIGH_PAIRED__condX1
        + 2.1635071398 aux_cap_helper__b2,r7,band_HIGH_PAIRED__condX2
        + 1.3908260184 aux_cap_helper__b2,r7,band_HIGH_PAIRED__condX3
        - aux_cap_b2,r7,band_HIGH_PAIRED
        - 0.6181448971 aux_cap_helper__b2,r7,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b2,r7,band_HIGH_PAIRED__Z3  = 0
 c4827: - aux_sv_b8,r3 + 349.915796675837 sv_function_helpervar_b8,r3__Z3
        + 207.605121921733 sv_function_helpervar_b8,r3__Z2
        + 27.8639631708 sv_function_helpervar_b8,r3__condX3
        + 45.0400196785 sv_function_helpervar_b8,r3__condX2
        + 1343.0085753241 sv_function_helpervar_b8,r3__condX1  = 0
 c4828: aux_cap_helper__b4,r6,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b4,r6,band_HIGH_PAIRED__Z3 <= 0
 c4829: aux_cap_b8,r2,band_HIGH_PAIRED + aux_cap_b8,r2,band_UNPAIRED
        - aux_c_b8,r2 + aux_cap_b8,r2,band_LOW_PAIRED  = 0
 c4830: 14 GlobalBidder_hat{W}__i8,k0 + GlobalBidder_W_i_8 <= 28
 c4831: aux_cap_helper__b4,r4,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b4,r4,band_UNPAIRED__Z2 <= 0
 c4832: - aux_cap_helper__b6,r6,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b6,r6,band_HIGH_PAIRED__Z2 <= 0
 c4833: X__b8,r9,band_LOW_PAIRED + X__b8,r9,band_HIGH_PAIRED
        + X__b8,r9,band_UNPAIRED - GlobalBidder_W_ir__b8,r9 >= 0
 c4834: - aux_cap_b4,r10,band_HIGH_PAIRED
        + 0.927217345653617 aux_cap_helper__b4,r10,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b4,r10,band_HIGH_PAIRED__Z2
        + 1.3908260184 aux_cap_helper__b4,r10,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b4,r10,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b4,r10,band_HIGH_PAIRED__condX1  = 0
 c4835: sv_function_helpervar_b9,r4__condX1
        - 0.159946187477637 sv_function_helpervar_b9,r4__Z1 <= 0
 c4836: aux_cap_helper__b2,r13,band_LOW_PAIRED__Z2
        + aux_cap_helper__b2,r13,band_LOW_PAIRED__Z1  = 1
 c4837: - aux_cap_helper__b7,r2,band_UNPAIRED__condX2
        + aux_cap_helper__b7,r2,band_UNPAIRED__Z2 <= 0
 c4838: aux_cap_helper__b0,r6,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b0,r6,band_UNPAIRED__Z2 <= 0
 c4839: aux_cap_helper__b2,r7,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b2,r7,band_HIGH_PAIRED__Z3 <= 0
 c4840: - aux_sv_b5,r0 + 1169.5974423155 sv_function_helpervar_b5,r0__condX1
        + 18.8855267756 sv_function_helpervar_b5,r0__condX3
        + 53.7549292076 sv_function_helpervar_b5,r0__condX2
        + 390.794872603881 sv_function_helpervar_b5,r0__Z3
        + 178.474755765569 sv_function_helpervar_b5,r0__Z2  = 0
 c4841: - aux_Omega_b9,r9 + 367.315797938751 aux_sv_b9,r9  = 0
 c4842: - 0.159946187445724 sv_function_helpervar_b9,r8__Z1
        + sv_function_helpervar_b9,r8__condX1 <= 0
 c4843: aux_cap_helper__b8,r12,band_LOW_PAIRED__Z1
        + aux_cap_helper__b8,r12,band_LOW_PAIRED__Z2  = 1
 c4844: 2 aux_cap_helper__b0,r1,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b0,r1,band_HIGH_PAIRED__condX3 <= 0
 c4845: 58.3052879626 sv_function_helpervar_b4,r4__condX2
        + 1289.3272633874 sv_function_helpervar_b4,r4__condX1
        + 21.0243786284 sv_function_helpervar_b4,r4__condX3
        + 196.897271643201 sv_function_helpervar_b4,r4__Z2
        + 427.511894937551 sv_function_helpervar_b4,r4__Z3 - aux_sv_b4,r4  = 0
 c4846: - 2 aux_cap_helper__b6,r3,band_UNPAIRED__Z2
        + aux_cap_helper__b6,r3,band_UNPAIRED__condX2 <= 0
 c4847: - aux_c_b7,r4 + aux_cap_b7,r4,band_UNPAIRED
        + aux_cap_b7,r4,band_LOW_PAIRED + aux_cap_b7,r4,band_HIGH_PAIRED  = 0
 c4848: sv_function_helpervar_b5,r8__Z1 + sv_function_helpervar_b5,r8__Z2
        + sv_function_helpervar_b5,r8__Z3  = 1
 c4849: - X__b6,r9,band_LOW_PAIRED
        + aux_cap_helper__b6,r9,band_LOW_PAIRED__condX1
        + aux_cap_helper__b6,r9,band_LOW_PAIRED__condX2  = 0
 c4850: - aux_cap_helper__b7,r11,band_UNPAIRED__Z1
        + aux_cap_helper__b7,r11,band_UNPAIRED__condX1 <= 0
 c4851: aux_cap_helper__b2,r3,band_LOW_PAIRED__Z2
        + aux_cap_helper__b2,r3,band_LOW_PAIRED__Z1  = 1
 c4852: - aux_cap_helper__b1,r10,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b1,r10,band_HIGH_PAIRED__condX1 <= 0
 c4853: aux_cap_helper__b0,r7,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b0,r7,band_HIGH_PAIRED__Z2 <= 0
 c4854: aux_cap_b6,r6,band_HIGH_PAIRED + aux_cap_b6,r6,band_UNPAIRED
        + aux_cap_b6,r6,band_LOW_PAIRED - aux_c_b6,r6  = 0
 c4855: aux_cap_helper__b1,r6,band_LOW_PAIRED__Z2
        - aux_cap_helper__b1,r6,band_LOW_PAIRED__condX2 <= 0
 c4856: - aux_cap_helper__b9,r8,band_UNPAIRED__condX1 <= 0
 c4857: - aux_cap_helper__b3,r13,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b3,r13,band_HIGH_PAIRED__Z2 <= 0
 c4858: - aux_cap_helper__b4,r7,band_LOW_PAIRED__Z1
        + aux_cap_helper__b4,r7,band_LOW_PAIRED__condX1 <= 0
 c4859: sv_function_helpervar_b5,r1__condX3
        - 15.9946187432626 sv_function_helpervar_b5,r1__Z3 <= 0
 c4860: sv_function_helpervar_b3,r11__Z2 + sv_function_helpervar_b3,r11__Z3
        + sv_function_helpervar_b3,r11__Z1  = 1
 c4861: aux_cap_helper__b4,r8,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b4,r8,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b4,r8,band_HIGH_PAIRED__condX1
        - X__b4,r8,band_HIGH_PAIRED  = 0
 c4862: aux_cap_helper__b8,r5,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b8,r5,band_HIGH_PAIRED__condX2 <= 0
 c4863: - aux_cap_helper__b1,r6,band_LOW_PAIRED__condX1 <= 0
 c4864: - aux_cap_helper__b6,r6,band_UNPAIRED__condX1 <= 0
 c4865: - aux_cap_helper__b3,r13,band_HIGH_PAIRED__condX1 <= 0
 c4866: - 2 aux_cap_helper__b9,r7,band_LOW_PAIRED__Z2
        + aux_cap_helper__b9,r7,band_LOW_PAIRED__condX2 <= 0
 c4867: - 8.28541025644105 sv_function_helpervar_b8,r3__Z2
        + sv_function_helpervar_b8,r3__condX2 <= 0
 c4868: aux_cap_helper__b7,r12,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b7,r12,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b7,r12,band_HIGH_PAIRED__Z1  = 1
 c4869: - aux_cap_helper__b8,r5,band_HIGH_PAIRED__condX1 <= 0
 c4870: 5.77782566925271 sv_function_helpervar_b3,r4__Z3
        - sv_function_helpervar_b3,r4__condX3 <= 0
 c4871: - sv_function_helpervar_b4,r2__condX2
        + 0.159946187453323 sv_function_helpervar_b4,r2__Z2 <= 0
 c4872: - aux_cap_helper__b5,r10,band_UNPAIRED__condX1 <= 0
 c4873: aux_cap_helper__b1,r12,band_UNPAIRED__Z1
        + aux_cap_helper__b1,r12,band_UNPAIRED__Z2  = 1
 c4874: - aux_Omega_b1,r3 + 487.243713086748 aux_sv_b1,r3  = 0
 c4875: aux_cap_helper__b4,r8,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b4,r8,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b4,r8,band_HIGH_PAIRED__Z1  = 1
 c4876: - aux_cap_helper__b7,r9,band_UNPAIRED__condX1 <= 0
 c4877: sv_function_helpervar_b2,r2__condX2
        - 6.10017409005981 sv_function_helpervar_b2,r2__Z2 <= 0
 c4878: - sv_function_helpervar_b8,r6__condX3
        + 6.94024412210506 sv_function_helpervar_b8,r6__Z3 <= 0
 c4879: aux_cap_helper__b9,r7,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b9,r7,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b9,r7,band_HIGH_PAIRED__Z1  = 1
 c4880: - aux_cap_helper__b5,r5,band_HIGH_PAIRED__condX1 <= 0
 c4881: - sv_function_helpervar_b4,r5__condX2
        + 0.15994618743266 sv_function_helpervar_b4,r5__Z2 <= 0
 c4882: 2.5999668471 sv_function_helpervar_b2,r2__condX3
        + 7.3781960991 sv_function_helpervar_b2,r2__condX2
        + 160.836769397 sv_function_helpervar_b2,r2__condX1 - aux_sv_b2,r2
        + 53.6931440111197 sv_function_helpervar_b2,r2__Z3
        + 24.5451137317034 sv_function_helpervar_b2,r2__Z2  = 0
 c4883: 1059.4718674279 sv_function_helpervar_b6,r2__condX1
        + 53.7845354943 sv_function_helpervar_b6,r2__condX2
        + 16.1900523394 sv_function_helpervar_b6,r2__condX3
        + 368.670307296291 sv_function_helpervar_b6,r2__Z3
        + 160.855854515929 sv_function_helpervar_b6,r2__Z2 - aux_sv_b6,r2  = 0
 c4884: 1.1250244065 aux_cap_helper__b8,r3,band_UNPAIRED__condX2
        + 0.8035888618 aux_cap_helper__b8,r3,band_UNPAIRED__condX1
        - 0.3214355447 aux_cap_helper__b8,r3,band_UNPAIRED__Z2
        - aux_cap_b8,r3,band_UNPAIRED  = 0
 c4885: 1580.59880088095 aux_sv_b1,r11 - aux_Omega_b1,r11  = 0
 c4886: - aux_cap_helper__b9,r8,band_UNPAIRED__condX2
        + aux_cap_helper__b9,r8,band_UNPAIRED__Z2 <= 0
 c4887: aux_cap_helper__b3,r8,band_UNPAIRED__Z1
        + aux_cap_helper__b3,r8,band_UNPAIRED__Z2  = 1
 c4888: - aux_cap_b9,r8,band_HIGH_PAIRED
        + 0.927217345653617 aux_cap_helper__b9,r8,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b9,r8,band_HIGH_PAIRED__Z2
        + 2.1635071398 aux_cap_helper__b9,r8,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b9,r8,band_HIGH_PAIRED__condX1
        + 1.3908260184 aux_cap_helper__b9,r8,band_HIGH_PAIRED__condX3  = 0
 c4889: - aux_cap_helper__b7,r3,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b7,r3,band_HIGH_PAIRED__condX1 <= 0
 c4890: aux_cap_helper__b1,r13,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b1,r13,band_HIGH_PAIRED__Z3 <= 0
 c4891: - 15.9946187432626 sv_function_helpervar_b5,r13__Z3
        + sv_function_helpervar_b5,r13__condX3 <= 0
 c4892: - sv_function_helpervar_b1,r6__condX2
        + 0.159946187453889 sv_function_helpervar_b1,r6__Z2 <= 0
 c4893: - aux_sv_b1,r3 + 19.1985015301478 sv_function_helpervar_b1,r3__Z2
        + 41.0959624291439 sv_function_helpervar_b1,r3__Z3
        + 125.5635947389 sv_function_helpervar_b1,r3__condX1
        + 5.5325903224 sv_function_helpervar_b1,r3__condX2
        + 2.0811417074 sv_function_helpervar_b1,r3__condX3  = 0
 c4894: - 0.6181448971 aux_cap_helper__b0,r0,band_HIGH_PAIRED__Z2
        + 0.927217345653617 aux_cap_helper__b0,r0,band_HIGH_PAIRED__Z3
        + 1.3908260184 aux_cap_helper__b0,r0,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b0,r0,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b0,r0,band_HIGH_PAIRED__condX1
        - aux_cap_b0,r0,band_HIGH_PAIRED  = 0
 c4895: sv_function_helpervar_b5,r4__Z3 + sv_function_helpervar_b5,r4__Z2
        + sv_function_helpervar_b5,r4__Z1  = 1
 c4896: aux_cap_helper__b8,r13,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b8,r13,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b8,r13,band_HIGH_PAIRED__condX2
        - X__b8,r13,band_HIGH_PAIRED  = 0
 c4897: - aux_cap_helper__b6,r6,band_UNPAIRED__condX2
        + aux_cap_helper__b6,r6,band_UNPAIRED__Z2 <= 0
 c4898: - 2 aux_cap_helper__b1,r9,band_UNPAIRED__Z2
        + aux_cap_helper__b1,r9,band_UNPAIRED__condX2 <= 0
 c4899: - 2 aux_cap_helper__b5,r0,band_UNPAIRED__Z2
        + aux_cap_helper__b5,r0,band_UNPAIRED__condX2 <= 0
 c4900: aux_cap_helper__b5,r7,band_UNPAIRED__condX1
        + aux_cap_helper__b5,r7,band_UNPAIRED__condX2 - X__b5,r7,band_UNPAIRED
         = 0
 c4901: - 2 aux_cap_helper__b8,r5,band_LOW_PAIRED__Z2
        + aux_cap_helper__b8,r5,band_LOW_PAIRED__condX2 <= 0
 c4902: - 2 aux_cap_helper__b6,r1,band_LOW_PAIRED__Z2
        + aux_cap_helper__b6,r1,band_LOW_PAIRED__condX2 <= 0
 c4903: - X__b3,r3,band_HIGH_PAIRED
        + aux_cap_helper__b3,r3,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b3,r3,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b3,r3,band_HIGH_PAIRED__condX1  = 0
 c4904: - 1.4943850124 aux_cap_helper__b0,r13,band_LOW_PAIRED__Z2
        - aux_cap_b0,r13,band_LOW_PAIRED
        + 3.735962531 aux_cap_helper__b0,r13,band_LOW_PAIRED__condX1
        + 5.2303475434 aux_cap_helper__b0,r13,band_LOW_PAIRED__condX2  = 0
 c4905: - aux_cap_helper__b3,r5,band_LOW_PAIRED__Z1
        + aux_cap_helper__b3,r5,band_LOW_PAIRED__condX1 <= 0
 c4906: 0.159946187432792 sv_function_helpervar_b1,r3__Z2
        - sv_function_helpervar_b1,r3__condX2 <= 0
 c4907: - aux_cap_helper__b1,r1,band_LOW_PAIRED__Z1
        + aux_cap_helper__b1,r1,band_LOW_PAIRED__condX1 <= 0
 c4908: sv_function_helpervar_b1,r9__Z3 + sv_function_helpervar_b1,r9__Z2
        + sv_function_helpervar_b1,r9__Z1  = 1
 c4909: aux_cap_helper__b6,r12,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b6,r12,band_HIGH_PAIRED__Z1 <= 0
 c4910: - 6.30423233120729 sv_function_helpervar_b0,r4__Z2
        + sv_function_helpervar_b0,r4__condX2 <= 0
 c4911: aux_cap_b9,r0,band_LOW_PAIRED + aux_cap_b9,r0,band_UNPAIRED
        + aux_cap_b9,r0,band_HIGH_PAIRED - aux_c_b9,r0  = 0
 c4912: aux_cap_helper__b2,r12,band_UNPAIRED__condX1
        + aux_cap_helper__b2,r12,band_UNPAIRED__condX2
        - X__b2,r12,band_UNPAIRED  = 0
 c4913: GlobalBidder_W_i_7 - 14 GlobalBidder_hat{W}__i7,k0 >= 0
 c4914: aux_cap_helper__b0,r8,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b0,r8,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b0,r8,band_HIGH_PAIRED__Z1  = 1
 c4915: - X__b3,r11,band_LOW_PAIRED
        + aux_cap_helper__b3,r11,band_LOW_PAIRED__condX2
        + aux_cap_helper__b3,r11,band_LOW_PAIRED__condX1  = 0
 c4916: sv_function_helpervar_b2,r3__condX1
        - 0.159946187533302 sv_function_helpervar_b2,r3__Z1 <= 0
 c4917: aux_cap_helper__b6,r1,band_LOW_PAIRED__Z1
        + aux_cap_helper__b6,r1,band_LOW_PAIRED__Z2  = 1
 c4918: sv_function_helpervar_b1,r5__condX3
        + sv_function_helpervar_b1,r5__condX2
        + sv_function_helpervar_b1,r5__condX1 - aux_c_b1,r5  = 0
 c4919: aux_cap_helper__b2,r1,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b2,r1,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b2,r1,band_HIGH_PAIRED__condX3
        - X__b2,r1,band_HIGH_PAIRED  = 0
 c4920: - 2 aux_cap_helper__b3,r1,band_UNPAIRED__Z2
        + aux_cap_helper__b3,r1,band_UNPAIRED__condX2 <= 0
 c4921: - sv_function_helpervar_b5,r0__condX3
        + 6.08900933282028 sv_function_helpervar_b5,r0__Z3 <= 0
 c4922: aux_cap_helper__b7,r3,band_UNPAIRED__Z2
        + aux_cap_helper__b7,r3,band_UNPAIRED__Z1  = 1
 c4923: aux_cap_helper__b7,r3,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b7,r3,band_LOW_PAIRED__Z2 <= 0
 c4924: aux_cap_b0,r1,band_LOW_PAIRED + aux_cap_b0,r1,band_UNPAIRED
        - aux_c_b0,r1 + aux_cap_b0,r1,band_HIGH_PAIRED  = 0
 c4925: aux_cap_helper__b8,r5,band_UNPAIRED__Z2
        - aux_cap_helper__b8,r5,band_UNPAIRED__condX2 <= 0
 c4926: 77.5579007117 sv_function_helpervar_b9,r6__condX2
        + 28.0728608207 sv_function_helpervar_b9,r6__condX3
        + 1719.0945881428 sv_function_helpervar_b9,r6__condX1
        + 262.557534708307 sv_function_helpervar_b9,r6__Z2
        + 569.365387598526 sv_function_helpervar_b9,r6__Z3 - aux_sv_b9,r6  = 0
 c4927: 0.159946187491808 sv_function_helpervar_b9,r12__Z2
        - sv_function_helpervar_b9,r12__condX2 <= 0
 c4928: 58.8805038326 sv_function_helpervar_b9,r3__condX2
        + 1719.0945881629 sv_function_helpervar_b9,r3__condX1
        + 34.8997302109 sv_function_helpervar_b9,r3__condX3 - aux_sv_b9,r3
        + 460.172214491696 sv_function_helpervar_b9,r3__Z3
        + 265.544913130648 sv_function_helpervar_b9,r3__Z2  = 0
 c4929: 0.8035888618 aux_cap_helper__b0,r2,band_UNPAIRED__condX1
        + 1.1250244065 aux_cap_helper__b0,r2,band_UNPAIRED__condX2
        - 0.3214355447 aux_cap_helper__b0,r2,band_UNPAIRED__Z2
        - aux_cap_b0,r2,band_UNPAIRED  = 0
 c4930: - sv_function_helpervar_b7,r2__condX3
        + 7.30447265021623 sv_function_helpervar_b7,r2__Z3 <= 0
 c4931: 7.89939018985145 sv_function_helpervar_b8,r13__Z3
        - sv_function_helpervar_b8,r13__condX3 <= 0
 c4932: 463.432993400844 sv_function_helpervar_b3,r2__Z3
        + 210.398472323986 sv_function_helpervar_b3,r2__Z2
        + 22.1215656848 sv_function_helpervar_b3,r2__condX3
        + 64.1556921212 sv_function_helpervar_b3,r2__condX2
        + 1379.5885614015 sv_function_helpervar_b3,r2__condX1 - aux_sv_b3,r2
         = 0
 c4933: - 2 aux_cap_helper__b8,r10,band_UNPAIRED__Z2
        + aux_cap_helper__b8,r10,band_UNPAIRED__condX2 <= 0
 c4934: - aux_cap_helper__b8,r5,band_UNPAIRED__condX1 <= 0
 c4935: - 0.159946187547474 sv_function_helpervar_b6,r10__Z1
        + sv_function_helpervar_b6,r10__condX1 <= 0
 c4936: - aux_c_b6,r11 + sv_function_helpervar_b6,r11__condX3
        + sv_function_helpervar_b6,r11__condX1
        + sv_function_helpervar_b6,r11__condX2  = 0
 c4937: - aux_cap_helper__b7,r9,band_UNPAIRED__condX2
        + aux_cap_helper__b7,r9,band_UNPAIRED__Z2 <= 0
 c4938: - aux_cap_helper__b2,r3,band_LOW_PAIRED__Z1
        + aux_cap_helper__b2,r3,band_LOW_PAIRED__condX1 <= 0
 c4939: - aux_cap_helper__b9,r1,band_UNPAIRED__condX1 <= 0
 c4940: - 5.89050642741838 sv_function_helpervar_b4,r11__Z2
        + sv_function_helpervar_b4,r11__condX2 <= 0
 c4941: aux_cap_helper__b4,r11,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b4,r11,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b4,r11,band_HIGH_PAIRED__condX1
        - X__b4,r11,band_HIGH_PAIRED  = 0
 c4942: - X__b2,r8,band_UNPAIRED + aux_cap_helper__b2,r8,band_UNPAIRED__condX2
        + aux_cap_helper__b2,r8,band_UNPAIRED__condX1  = 0
 c4943: 0.927217345653617 aux_cap_helper__b5,r11,band_HIGH_PAIRED__Z3
        - 0.6181448971 aux_cap_helper__b5,r11,band_HIGH_PAIRED__Z2
        - aux_cap_b5,r11,band_HIGH_PAIRED
        + 1.3908260184 aux_cap_helper__b5,r11,band_HIGH_PAIRED__condX3
        + 2.1635071398 aux_cap_helper__b5,r11,band_HIGH_PAIRED__condX2
        + 1.5453622427 aux_cap_helper__b5,r11,band_HIGH_PAIRED__condX1  = 0
 c4944: 0.8035888618 aux_cap_helper__b9,r0,band_UNPAIRED__condX1
        + 1.1250244065 aux_cap_helper__b9,r0,band_UNPAIRED__condX2
        - aux_cap_b9,r0,band_UNPAIRED
        - 0.3214355447 aux_cap_helper__b9,r0,band_UNPAIRED__Z2  = 0
Bounds
 0 <= aux_cap_helper__b0,r11,band_LOW_PAIRED__condX2 <= 2
 0 <= X__b2,r2,band_LOW_PAIRED <= 2
 0 <= aux_cap_helper__b0,r11,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b5,r5,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b2,r10,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b4,r10,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b2,r10,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b4,r10,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_sv_b0,r2 <= 536870910
 0 <= aux_sv_b0,r1 <= 536870910
 0 <= aux_sv_b0,r4 <= 536870910
 0 <= aux_sv_b0,r3 <= 536870910
 0 <= aux_cap_b5,r3,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_b8,r11,band_LOW_PAIRED <= 536870910
 0 <= X__b2,r12,band_HIGH_PAIRED <= 3
 0 <= aux_sv_b0,r0 <= 536870910
 0 <= aux_sv_b0,r9 <= 536870910
 0 <= aux_cap_helper__b2,r12,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b2,r12,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_sv_b0,r6 <= 536870910
 0 <= aux_sv_b0,r5 <= 536870910
 0 <= aux_sv_b0,r8 <= 536870910
 0 <= aux_sv_b0,r7 <= 536870910
 0 <= X__b4,r7,band_LOW_PAIRED <= 2
 0 <= aux_cap_helper__b3,r5,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_b1,r1,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b3,r5,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b3,r11,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b3,r11,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_Omega_b0,r12 <= 536870910
 0 <= X__b6,r0,band_LOW_PAIRED <= 2
 0 <= aux_Omega_b0,r13 <= 536870910
 0 <= aux_Omega_b0,r10 <= 536870910
 0 <= aux_Omega_b0,r11 <= 536870910
 0 <= aux_cap_b1,r4,band_LOW_PAIRED <= 536870910
 0 <= X__b7,r1,band_HIGH_PAIRED <= 3
 0 <= sv_function_helpervar_b0,r10__Z2 <= 1
 0 <= aux_cap_helper__b1,r13,band_LOW_PAIRED__Z2 <= 1
 0 <= sv_function_helpervar_b0,r10__Z3 <= 1
 0 <= aux_cap_helper__b1,r13,band_LOW_PAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b0,r10__Z1 <= 1
 0 <= X__b5,r7,band_HIGH_PAIRED <= 3
 0 <= aux_cap_helper__b4,r6,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b4,r4,band_HIGH_PAIRED__Z1 <= 1
 0 <= X__b9,r12,band_LOW_PAIRED <= 2
 0 <= aux_cap_helper__b4,r6,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b4,r4,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b4,r4,band_HIGH_PAIRED__Z2 <= 1
 0 <= X__b0,r9,band_LOW_PAIRED <= 2
 0 <= X__b2,r9,band_HIGH_PAIRED <= 3
 0 <= sv_function_helpervar_b3,r3__condX3 <= 536870910
 0 <= sv_function_helpervar_b3,r3__condX2 <= 536870910
 0 <= sv_function_helpervar_b3,r3__condX1 <= 536870910
 0 <= aux_cap_b8,r1,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b2,r12,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b7,r1,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b7,r1,band_LOW_PAIRED__condX2 <= 2
 0 <= sv_function_helpervar_b3,r9__condX3 <= 536870910
 0 <= X__b4,r0,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b2,r12,band_HIGH_PAIRED__Z2 <= 1
 0 <= sv_function_helpervar_b3,r9__condX1 <= 536870910
 0 <= aux_cap_helper__b2,r12,band_HIGH_PAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b3,r9__condX2 <= 536870910
 0 <= aux_cap_helper__b3,r2,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b3,r2,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_b5,r2,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_helper__b9,r5,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b9,r5,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b9,r5,band_HIGH_PAIRED__Z3 <= 1
 0 <= X__b8,r5,band_HIGH_PAIRED <= 3
 0 <= X__b4,r13,band_LOW_PAIRED <= 2
 0 <= aux_cap_helper__b7,r8,band_HIGH_PAIRED__condX2 <= 3
 0 <= sv_function_helpervar_b1,r4__condX1 <= 536870910
 0 <= aux_cap_helper__b7,r8,band_HIGH_PAIRED__condX1 <= 3
 0 <= sv_function_helpervar_b1,r4__condX2 <= 536870910
 0 <= sv_function_helpervar_b1,r4__condX3 <= 536870910
 0 <= aux_cap_helper__b7,r8,band_HIGH_PAIRED__condX3 <= 3
 0 <= X__b3,r10,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b3,r6,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_b2,r5,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b3,r6,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b3,r6,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_b7,r0,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b1,r9,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_b6,r10,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_helper__b1,r9,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_b6,r8,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_b1,r12,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b4,r5,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b4,r5,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_b5,r13,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_helper__b7,r9,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_b1,r3,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b1,r11,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b1,r11,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b1,r11,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b7,r9,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b5,r7,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b7,r9,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b5,r7,band_UNPAIRED__condX2 <= 2
 0 <= sv_function_helpervar_b5,r3__Z2 <= 1
 0 <= sv_function_helpervar_b5,r3__Z1 <= 1
 0 <= sv_function_helpervar_b5,r3__Z3 <= 1
 0 <= aux_cap_helper__b5,r5,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_c_b7,r12 <= 536870910
 0 <= aux_c_b7,r11 <= 536870910
 0 <= aux_c_b7,r10 <= 536870910
 0 <= sv_function_helpervar_b9,r12__Z3 <= 1
 0 <= aux_c_b7,r13 <= 536870910
 0 <= sv_function_helpervar_b9,r12__Z1 <= 1
 0 <= sv_function_helpervar_b9,r12__Z2 <= 1
 0 <= aux_cap_helper__b7,r8,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b7,r8,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b4,r7,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b4,r7,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b2,r8,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b9,r0,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b9,r0,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b2,r8,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b2,r8,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b6,r7,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b6,r7,band_LOW_PAIRED__condX1 <= 2
 0 <= sv_function_helpervar_b3,r2__Z1 <= 1
 0 <= sv_function_helpervar_b3,r2__Z3 <= 1
 0 <= aux_cap_b4,r11,band_HIGH_PAIRED <= 536870910
 0 <= sv_function_helpervar_b3,r2__Z2 <= 1
 0 <= aux_cap_helper__b7,r10,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b8,r10,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_b4,r3,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_helper__b4,r9,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b4,r9,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b4,r9,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b7,r10,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_b9,r12,band_HIGH_PAIRED <= 536870910
 0 <= X__b5,r1,band_LOW_PAIRED <= 2
 0 <= sv_function_helpervar_b0,r4__Z1 <= 1
 0 <= sv_function_helpervar_b0,r4__Z2 <= 1
 0 <= aux_cap_helper__b6,r0,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b6,r0,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b2,r3,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b2,r3,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_b9,r4,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b2,r3,band_HIGH_PAIRED__condX1 <= 3
 0 <= sv_function_helpervar_b0,r4__Z3 <= 1
 0 <= aux_cap_helper__b2,r12,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b2,r12,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b9,r3,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b9,r3,band_LOW_PAIRED__condX1 <= 2
 0 <= X__b7,r11,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b5,r10,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b6,r4,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b6,r4,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b5,r10,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_sv_b4,r10 <= 536870910
 0 <= aux_sv_b4,r11 <= 536870910
 0 <= aux_sv_b4,r12 <= 536870910
 0 <= aux_sv_b4,r13 <= 536870910
 0 <= aux_cap_b9,r5,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b0,r8,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b6,r8,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b0,r8,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b0,r8,band_HIGH_PAIRED__condX1 <= 3
 0 <= X__b9,r8,band_HIGH_PAIRED <= 3
 0 <= aux_cap_helper__b5,r2,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b5,r2,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_b7,r7,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_helper__b5,r2,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b6,r8,band_UNPAIRED__condX2 <= 2
 0 <= X__b7,r6,band_LOW_PAIRED <= 2
 0 <= X__b5,r2,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b3,r4,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b3,r4,band_LOW_PAIRED__condX1 <= 2
 0 <= X__b1,r3,band_LOW_PAIRED <= 2
 0 <= X__b4,r3,band_HIGH_PAIRED <= 3
 0 <= aux_cap_helper__b6,r4,band_HIGH_PAIRED__condX1 <= 3
 0 <= X__b2,r12,band_LOW_PAIRED <= 2
 0 <= X__b0,r2,band_HIGH_PAIRED <= 3
 0 <= aux_cap_b6,r6,band_HIGH_PAIRED <= 536870910
 0 <= X__b5,r6,band_UNPAIRED <= 2
 0 <= X__b8,r5,band_LOW_PAIRED <= 2
 0 <= sv_function_helpervar_b1,r8__Z3 <= 1
 0 <= sv_function_helpervar_b1,r8__Z2 <= 1
 0 <= aux_cap_b2,r3,band_LOW_PAIRED <= 536870910
 0 <= sv_function_helpervar_b1,r8__Z1 <= 1
 0 <= aux_cap_helper__b0,r8,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b0,r8,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b2,r4,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b7,r11,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b8,r1,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b7,r11,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b8,r1,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b7,r11,band_HIGH_PAIRED__Z2 <= 1
 0 <= sv_function_helpervar_b7,r8__Z2 <= 1
 0 <= aux_cap_helper__b8,r1,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b2,r4,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b7,r11,band_HIGH_PAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b7,r8__Z3 <= 1
 0 <= aux_Omega_b2,r9 <= 536870910
 0 <= aux_cap_helper__b7,r11,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b7,r11,band_LOW_PAIRED__condX1 <= 2
 0 <= sv_function_helpervar_b7,r8__Z1 <= 1
 0 <= aux_Omega_b2,r7 <= 536870910
 0 <= X__b5,r4,band_UNPAIRED <= 2
 0 <= aux_Omega_b2,r8 <= 536870910
 0 <= aux_Omega_b2,r5 <= 536870910
 0 <= aux_Omega_b2,r6 <= 536870910
 0 <= aux_Omega_b2,r3 <= 536870910
 0 <= aux_Omega_b2,r4 <= 536870910
 0 <= aux_Omega_b2,r1 <= 536870910
 0 <= aux_cap_helper__b5,r2,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_Omega_b2,r2 <= 536870910
 0 <= aux_cap_helper__b5,r2,band_HIGH_PAIRED__condX1 <= 3
 0 <= X__b7,r11,band_LOW_PAIRED <= 2
 0 <= aux_Omega_b2,r0 <= 536870910
 0 <= aux_cap_helper__b5,r2,band_HIGH_PAIRED__condX2 <= 3
 0 <= X__b5,r8,band_UNPAIRED <= 2
 0 <= aux_cap_b4,r12,band_UNPAIRED <= 536870910
 0 <= X__b5,r11,band_HIGH_PAIRED <= 3
 0 <= sv_function_helpervar_b6,r10__Z1 <= 1
 0 <= aux_cap_b1,r1,band_HIGH_PAIRED <= 536870910
 0 <= sv_function_helpervar_b6,r10__Z3 <= 1
 0 <= sv_function_helpervar_b6,r10__Z2 <= 1
 0 <= aux_cap_helper__b7,r9,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b7,r9,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b8,r10,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b8,r10,band_HIGH_PAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b9,r0__condX1 <= 536870910
 0 <= sv_function_helpervar_b9,r0__condX2 <= 536870910
 0 <= aux_cap_helper__b6,r4,band_UNPAIRED__Z2 <= 1
 0 <= sv_function_helpervar_b9,r0__condX3 <= 536870910
 0 <= aux_cap_helper__b6,r4,band_UNPAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b8,r1__Z3 <= 1
 0 <= sv_function_helpervar_b8,r1__Z2 <= 1
 0 <= sv_function_helpervar_b8,r1__Z1 <= 1
 0 <= aux_cap_helper__b9,r3,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b9,r3,band_HIGH_PAIRED__condX3 <= 3
 0 <= sv_function_helpervar_b2,r9__Z2 <= 1
 0 <= sv_function_helpervar_b2,r9__Z3 <= 1
 0 <= aux_cap_helper__b9,r3,band_HIGH_PAIRED__condX1 <= 3
 0 <= sv_function_helpervar_b2,r9__Z1 <= 1
 0 <= aux_cap_helper__b1,r1,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b1,r1,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_b7,r2,band_LOW_PAIRED <= 536870910
 0 <= X__b7,r13,band_HIGH_PAIRED <= 3
 0 <= X__b7,r13,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b3,r7,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b8,r13,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b8,r13,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b3,r7,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b1,r13,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b1,r13,band_HIGH_PAIRED__condX1 <= 3
 0 <= sv_function_helpervar_b7,r11__condX3 <= 536870910
 0 <= aux_cap_b1,r12,band_UNPAIRED <= 536870910
 0 <= sv_function_helpervar_b7,r11__condX2 <= 536870910
 0 <= sv_function_helpervar_b7,r11__condX1 <= 536870910
 0 <= aux_cap_helper__b1,r13,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_b8,r0,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_b1,r13,band_HIGH_PAIRED <= 536870910
 0 <= sv_function_helpervar_b4,r9__condX1 <= 536870910
 0 <= aux_cap_b2,r3,band_UNPAIRED <= 536870910
 0 <= sv_function_helpervar_b4,r9__condX3 <= 536870910
 0 <= sv_function_helpervar_b4,r9__condX2 <= 536870910
 0 <= X__b0,r9,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b4,r9,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b4,r9,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b3,r3,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b3,r3,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b8,r5,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_b8,r6,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_helper__b8,r5,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b7,r1,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_b5,r11,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b7,r1,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b4,r4,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b4,r4,band_UNPAIRED__condX1 <= 2
 0 <= sv_function_helpervar_b0,r4__condX3 <= 536870910
 0 <= sv_function_helpervar_b0,r4__condX2 <= 536870910
 0 <= aux_cap_helper__b6,r8,band_HIGH_PAIRED__Z2 <= 1
 0 <= sv_function_helpervar_b0,r4__condX1 <= 536870910
 0 <= aux_cap_helper__b6,r8,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_b3,r4,band_LOW_PAIRED <= 536870910
 0 <= X__b5,r6,band_HIGH_PAIRED <= 3
 0 <= aux_cap_helper__b6,r8,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b2,r6,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b2,r6,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b2,r8,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b2,r0,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_b1,r7,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b2,r0,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b2,r8,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b2,r0,band_HIGH_PAIRED__Z1 <= 1
 0 <= X__b8,r4,band_HIGH_PAIRED <= 3
 0 <= aux_cap_helper__b8,r2,band_UNPAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b2,r3__condX3 <= 536870910
 0 <= sv_function_helpervar_b2,r3__condX2 <= 536870910
 0 <= aux_cap_helper__b4,r8,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_b4,r0,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b4,r8,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b9,r4,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b9,r4,band_UNPAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b2,r3__condX1 <= 536870910
 0 <= aux_cap_helper__b6,r6,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b6,r6,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b6,r6,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b8,r2,band_UNPAIRED__Z2 <= 1
 0 <= X__b2,r11,band_HIGH_PAIRED <= 3
 0 <= aux_cap_helper__b1,r7,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b0,r13,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b1,r7,band_HIGH_PAIRED__Z2 <= 1
 0 <= sv_function_helpervar_b6,r2__condX1 <= 536870910
 0 <= aux_cap_b9,r7,band_LOW_PAIRED <= 536870910
 0 <= sv_function_helpervar_b6,r2__condX2 <= 536870910
 0 <= aux_cap_b8,r11,band_UNPAIRED <= 536870910
 0 <= sv_function_helpervar_b6,r2__condX3 <= 536870910
 0 <= aux_cap_helper__b0,r13,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b1,r7,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b9,r9,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b9,r9,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b2,r2,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b2,r2,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b2,r1,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b2,r1,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_b4,r10,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b0,r9,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b9,r11,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b0,r9,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b2,r8,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b9,r11,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b0,r9,band_HIGH_PAIRED__Z3 <= 1
 0 <= X__b8,r10,band_HIGH_PAIRED <= 3
 0 <= aux_cap_helper__b2,r1,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b2,r8,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b9,r12,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b9,r12,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_Omega_b5,r1 <= 536870910
 0 <= X__b6,r7,band_LOW_PAIRED <= 2
 0 <= aux_Omega_b5,r0 <= 536870910
 0 <= X__b4,r2,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b8,r12,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b8,r12,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b3,r7,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_b1,r5,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b3,r7,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b1,r13,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b1,r13,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_b3,r7,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b3,r0,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b3,r0,band_LOW_PAIRED__condX2 <= 2
 0 <= X__b1,r6,band_HIGH_PAIRED <= 3
 0 <= X__b2,r8,band_HIGH_PAIRED <= 3
 0 <= aux_cap_helper__b6,r3,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b1,r13,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b6,r3,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b1,r13,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b6,r3,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b1,r13,band_HIGH_PAIRED__Z3 <= 1
 0 <= sv_function_helpervar_b4,r6__Z3 <= 1
 0 <= sv_function_helpervar_b7,r7__condX2 <= 536870910
 0 <= sv_function_helpervar_b7,r7__condX1 <= 536870910
 0 <= sv_function_helpervar_b4,r6__Z1 <= 1
 0 <= sv_function_helpervar_b4,r6__Z2 <= 1
 0 <= X__b0,r4,band_LOW_PAIRED <= 2
 0 <= X__b2,r7,band_LOW_PAIRED <= 2
 0 <= aux_Omega_b5,r9 <= 536870910
 0 <= aux_Omega_b5,r8 <= 536870910
 0 <= X__b7,r2,band_HIGH_PAIRED <= 3
 0 <= aux_Omega_b5,r7 <= 536870910
 0 <= aux_Omega_b5,r6 <= 536870910
 0 <= aux_Omega_b5,r5 <= 536870910
 0 <= aux_cap_b4,r8,band_LOW_PAIRED <= 536870910
 0 <= aux_Omega_b5,r4 <= 536870910
 0 <= aux_Omega_b5,r3 <= 536870910
 0 <= aux_Omega_b5,r2 <= 536870910
 0 <= sv_function_helpervar_b7,r7__condX3 <= 536870910
 0 <= X__b4,r4,band_UNPAIRED <= 2
 0 <= aux_cap_b5,r2,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b3,r5,band_HIGH_PAIRED__condX1 <= 3
 0 <= X__b0,r11,band_LOW_PAIRED <= 2
 0 <= aux_cap_helper__b0,r2,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b0,r2,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b3,r5,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_b9,r13,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b3,r5,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_b3,r9,band_UNPAIRED <= 536870910
 0 <= sv_function_helpervar_b9,r7__condX3 <= 536870910
 0 <= sv_function_helpervar_b9,r7__condX2 <= 536870910
 0 <= sv_function_helpervar_b9,r7__condX1 <= 536870910
 0 <= X__b3,r0,band_UNPAIRED <= 2
 0 <= sv_function_helpervar_b9,r9__Z3 <= 1
 0 <= sv_function_helpervar_b9,r9__Z2 <= 1
 0 <= sv_function_helpervar_b9,r9__Z1 <= 1
 0 <= sv_function_helpervar_b4,r2__condX1 <= 536870910
 0 <= sv_function_helpervar_b4,r2__condX2 <= 536870910
 0 <= sv_function_helpervar_b4,r2__condX3 <= 536870910
 0 <= aux_cap_b8,r13,band_UNPAIRED <= 536870910
 0 <= sv_function_helpervar_b8,r1__condX2 <= 536870910
 0 <= aux_cap_helper__b9,r11,band_UNPAIRED__condX1 <= 2
 0 <= sv_function_helpervar_b8,r1__condX3 <= 536870910
 0 <= aux_cap_helper__b9,r11,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_b2,r5,band_UNPAIRED <= 536870910
 0 <= X__b1,r5,band_HIGH_PAIRED <= 3
 0 <= aux_cap_helper__b6,r3,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b6,r3,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b1,r4,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b1,r4,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b0,r12,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b0,r12,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b0,r12,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b0,r2,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b5,r0,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b7,r11,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b5,r0,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b7,r11,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b7,r11,band_HIGH_PAIRED__condX2 <= 3
 0 <= sv_function_helpervar_b8,r1__condX1 <= 536870910
 0 <= aux_cap_helper__b0,r2,band_UNPAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b0,r10__condX1 <= 536870910
 0 <= aux_cap_b8,r2,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b7,r12,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b7,r12,band_UNPAIRED__condX2 <= 2
 0 <= X__b4,r6,band_UNPAIRED <= 2
 0 <= sv_function_helpervar_b0,r10__condX3 <= 536870910
 0 <= aux_cap_b1,r2,band_HIGH_PAIRED <= 536870910
 0 <= sv_function_helpervar_b0,r10__condX2 <= 536870910
 0 <= sv_function_helpervar_b8,r10__condX3 <= 536870910
 0 <= sv_function_helpervar_b8,r10__condX2 <= 536870910
 0 <= aux_cap_helper__b9,r7,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b5,r11,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b9,r7,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b5,r11,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b1,r3,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b5,r11,band_HIGH_PAIRED__Z3 <= 1
 0 <= sv_function_helpervar_b1,r13__Z1 <= 1
 0 <= aux_cap_helper__b5,r11,band_HIGH_PAIRED__Z2 <= 1
 0 <= sv_function_helpervar_b8,r10__condX1 <= 536870910
 0 <= aux_cap_helper__b5,r11,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b1,r3,band_UNPAIRED__condX1 <= 2
 0 <= sv_function_helpervar_b3,r12__Z1 <= 1
 0 <= sv_function_helpervar_b3,r12__Z2 <= 1
 0 <= aux_cap_b2,r10,band_LOW_PAIRED <= 536870910
 0 <= sv_function_helpervar_b3,r12__Z3 <= 1
 0 <= sv_function_helpervar_b1,r13__Z3 <= 1
 0 <= sv_function_helpervar_b1,r13__Z2 <= 1
 0 <= X__b0,r7,band_UNPAIRED <= 2
 0 <= aux_cap_b8,r1,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_helper__b0,r1,band_HIGH_PAIRED__Z3 <= 1
 0 <= GlobalBidder_hat{W}__i8,k0 <= 1
 0 <= GlobalBidder_hat{W}__i8,k1 <= 1
 0 <= aux_cap_helper__b0,r1,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_b2,r1,band_UNPAIRED <= 536870910
 0 <= GlobalBidder_hat{W}__i8,k2 <= 1
 0 <= aux_cap_helper__b0,r1,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_b1,r10,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b5,r13,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b5,r13,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b1,r1,band_UNPAIRED__condX1 <= 2
 0 <= sv_function_helpervar_b0,r0__Z3 <= 1
 0 <= aux_cap_helper__b1,r1,band_UNPAIRED__condX2 <= 2
 0 <= sv_function_helpervar_b0,r0__Z2 <= 1
 0 <= sv_function_helpervar_b0,r0__Z1 <= 1
 0 <= aux_cap_helper__b3,r12,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b3,r12,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b1,r6,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b1,r6,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_b9,r3,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_b2,r9,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b1,r11,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b1,r11,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b7,r3,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b7,r3,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b2,r4,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b2,r4,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b7,r1,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b7,r1,band_HIGH_PAIRED__Z3 <= 1
 0 <= sv_function_helpervar_b6,r13__condX1 <= 536870910
 0 <= sv_function_helpervar_b6,r13__condX2 <= 536870910
 0 <= sv_function_helpervar_b6,r13__condX3 <= 536870910
 0 <= aux_cap_b3,r9,band_LOW_PAIRED <= 536870910
 0 <= aux_sv_b8,r11 <= 536870910
 0 <= aux_sv_b8,r10 <= 536870910
 0 <= aux_sv_b8,r13 <= 536870910
 0 <= aux_sv_b8,r12 <= 536870910
 0 <= aux_cap_helper__b5,r6,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b5,r6,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b7,r1,band_HIGH_PAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b2,r5__Z2 <= 1
 0 <= sv_function_helpervar_b2,r5__Z3 <= 1
 0 <= sv_function_helpervar_b7,r4__Z1 <= 1
 0 <= sv_function_helpervar_b2,r5__Z1 <= 1
 0 <= aux_cap_b2,r7,band_UNPAIRED <= 536870910
 0 <= aux_cap_b2,r4,band_HIGH_PAIRED <= 536870910
 0 <= X__b5,r8,band_LOW_PAIRED <= 2
 0 <= X__b4,r4,band_HIGH_PAIRED <= 3
 0 <= aux_cap_b3,r12,band_LOW_PAIRED <= 536870910
 0 <= X__b1,r8,band_LOW_PAIRED <= 2
 0 <= aux_cap_helper__b8,r7,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b8,r7,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_b9,r0,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_helper__b5,r9,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b8,r7,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b5,r9,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b9,r13,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b9,r13,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b2,r1,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b2,r1,band_LOW_PAIRED__Z2 <= 1
 0 <= X__b3,r2,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b9,r1,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b9,r1,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b9,r1,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b0,r0,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b0,r0,band_UNPAIRED__condX1 <= 2
 0 <= sv_function_helpervar_b5,r8__condX1 <= 536870910
 0 <= sv_function_helpervar_b5,r8__condX3 <= 536870910
 0 <= sv_function_helpervar_b5,r8__condX2 <= 536870910
 0 <= aux_cap_helper__b9,r12,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b9,r12,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b4,r10,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b4,r10,band_UNPAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b7,r4__Z2 <= 1
 0 <= sv_function_helpervar_b7,r4__Z3 <= 1
 0 <= sv_function_helpervar_b7,r2__condX1 <= 536870910
 0 <= sv_function_helpervar_b8,r9__Z3 <= 1
 0 <= sv_function_helpervar_b8,r9__Z2 <= 1
 0 <= sv_function_helpervar_b7,r2__condX3 <= 536870910
 0 <= sv_function_helpervar_b7,r2__condX2 <= 536870910
 0 <= sv_function_helpervar_b8,r9__Z1 <= 1
 0 <= aux_cap_helper__b3,r10,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b3,r10,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b3,r10,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b4,r2,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_b6,r2,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b4,r2,band_UNPAIRED__condX2 <= 2
 0 <= aux_c_b7,r0 <= 536870910
 0 <= aux_cap_b6,r5,band_HIGH_PAIRED <= 536870910
 0 <= aux_c_b7,r1 <= 536870910
 0 <= aux_c_b7,r2 <= 536870910
 0 <= aux_c_b7,r3 <= 536870910
 0 <= aux_cap_helper__b3,r9,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_c_b7,r4 <= 536870910
 0 <= aux_c_b7,r5 <= 536870910
 0 <= X__b1,r1,band_LOW_PAIRED <= 2
 0 <= aux_cap_helper__b3,r9,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_c_b7,r6 <= 536870910
 0 <= aux_cap_helper__b3,r9,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_c_b7,r7 <= 536870910
 0 <= aux_c_b7,r8 <= 536870910
 0 <= aux_cap_b0,r0,band_LOW_PAIRED <= 536870910
 0 <= aux_c_b7,r9 <= 536870910
 0 <= aux_cap_b7,r8,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b9,r2,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b9,r2,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b9,r2,band_HIGH_PAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b5,r9__condX3 <= 536870910
 0 <= sv_function_helpervar_b5,r9__condX2 <= 536870910
 0 <= sv_function_helpervar_b5,r9__condX1 <= 536870910
 0 <= X__b5,r12,band_HIGH_PAIRED <= 3
 0 <= aux_cap_helper__b5,r9,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_b3,r3,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b5,r9,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b6,r8,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b6,r8,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b7,r13,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b6,r8,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b7,r13,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b7,r13,band_HIGH_PAIRED__condX1 <= 3
 0 <= X__b7,r8,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b7,r5,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b7,r5,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b5,r1,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b5,r1,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b5,r2,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_b4,r11,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_helper__b5,r2,band_LOW_PAIRED__condX1 <= 2
 0 <= X__b3,r6,band_LOW_PAIRED <= 2
 0 <= aux_cap_helper__b4,r7,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b4,r7,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b1,r7,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b5,r12,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b5,r12,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b1,r7,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b4,r7,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b5,r12,band_HIGH_PAIRED__condX1 <= 3
 0 <= X__b7,r12,band_HIGH_PAIRED <= 3
 0 <= sv_function_helpervar_b7,r10__Z3 <= 1
 0 <= sv_function_helpervar_b7,r10__Z2 <= 1
 0 <= sv_function_helpervar_b7,r10__Z1 <= 1
 0 <= sv_function_helpervar_b0,r5__condX1 <= 536870910
 0 <= sv_function_helpervar_b0,r5__condX3 <= 536870910
 0 <= sv_function_helpervar_b0,r5__condX2 <= 536870910
 0 <= sv_function_helpervar_b8,r6__condX2 <= 536870910
 0 <= aux_cap_b0,r7,band_UNPAIRED <= 536870910
 0 <= sv_function_helpervar_b8,r6__condX1 <= 536870910
 0 <= sv_function_helpervar_b8,r6__condX3 <= 536870910
 0 <= sv_function_helpervar_b7,r1__Z1 <= 1
 0 <= X__b7,r4,band_LOW_PAIRED <= 2
 0 <= sv_function_helpervar_b7,r1__Z2 <= 1
 0 <= sv_function_helpervar_b7,r1__Z3 <= 1
 0 <= X__b7,r1,band_LOW_PAIRED <= 2
 0 <= aux_cap_b7,r11,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_b0,r9,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b4,r7,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b4,r7,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b4,r7,band_HIGH_PAIRED__condX2 <= 3
 0 <= sv_function_helpervar_b1,r5__Z2 <= 1
 0 <= sv_function_helpervar_b1,r5__Z1 <= 1
 0 <= aux_cap_helper__b9,r0,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b9,r0,band_UNPAIRED__condX2 <= 2
 0 <= X__b8,r10,band_LOW_PAIRED <= 2
 0 <= sv_function_helpervar_b1,r5__Z3 <= 1
 0 <= aux_cap_helper__b8,r6,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b8,r6,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b0,r10,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b2,r7,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b0,r10,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b2,r7,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_b6,r12,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_b8,r8,band_LOW_PAIRED <= 536870910
 0 <= X__b4,r5,band_HIGH_PAIRED <= 3
 0 <= aux_cap_b2,r8,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_b2,r5,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_helper__b3,r10,band_UNPAIRED__condX2 <= 2
 0 <= sv_function_helpervar_b0,r8__Z3 <= 1
 0 <= sv_function_helpervar_b0,r8__Z1 <= 1
 0 <= sv_function_helpervar_b0,r8__Z2 <= 1
 0 <= sv_function_helpervar_b4,r8__condX3 <= 536870910
 0 <= sv_function_helpervar_b4,r8__condX2 <= 536870910
 0 <= sv_function_helpervar_b4,r8__condX1 <= 536870910
 0 <= aux_cap_helper__b3,r10,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b7,r7,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b7,r7,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b9,r0,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b9,r0,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b9,r0,band_HIGH_PAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b1,r10__Z3 <= 1
 0 <= aux_cap_b5,r5,band_HIGH_PAIRED <= 536870910
 0 <= sv_function_helpervar_b1,r10__Z2 <= 1
 0 <= sv_function_helpervar_b1,r10__Z1 <= 1
 0 <= aux_cap_helper__b0,r10,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b0,r10,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b0,r4,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b0,r4,band_UNPAIRED__condX2 <= 2
 0 <= X__b0,r3,band_HIGH_PAIRED <= 3
 0 <= X__b3,r11,band_LOW_PAIRED <= 2
 0 <= X__b6,r2,band_LOW_PAIRED <= 2
 0 <= X__b6,r5,band_LOW_PAIRED <= 2
 0 <= aux_cap_b1,r9,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b7,r6,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b9,r0,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b9,r0,band_LOW_PAIRED__condX2 <= 2
 0 <= X__b4,r2,band_HIGH_PAIRED <= 3
 0 <= X__b0,r12,band_UNPAIRED <= 2
 0 <= aux_cap_b0,r3,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b7,r4,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b7,r6,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b7,r4,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b7,r6,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b3,r6,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b3,r6,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b3,r13,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_b2,r3,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b3,r13,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_b5,r11,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_helper__b4,r9,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b4,r9,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b4,r9,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b0,r7,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b5,r2,band_LOW_PAIRED__Z2 <= 1
 0 <= sv_function_helpervar_b9,r2__Z2 <= 1
 0 <= aux_cap_helper__b0,r7,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b5,r2,band_LOW_PAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b9,r2__Z3 <= 1
 0 <= sv_function_helpervar_b9,r2__Z1 <= 1
 0 <= aux_cap_helper__b4,r8,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b0,r5,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b0,r5,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_b9,r4,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b7,r0,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b7,r0,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b4,r8,band_LOW_PAIRED__condX1 <= 2
 0 <= sv_function_helpervar_b6,r8__Z2 <= 1
 0 <= sv_function_helpervar_b6,r8__Z1 <= 1
 0 <= sv_function_helpervar_b6,r8__Z3 <= 1
 0 <= aux_cap_b5,r4,band_LOW_PAIRED <= 536870910
 0 <= X__b9,r9,band_LOW_PAIRED <= 2
 0 <= aux_cap_helper__b7,r6,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b7,r6,band_HIGH_PAIRED__Z1 <= 1
 0 <= X__b2,r11,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b7,r6,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b1,r5,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b1,r10,band_LOW_PAIRED__Z1 <= 1
 0 <= X__b0,r2,band_LOW_PAIRED <= 2
 0 <= aux_cap_helper__b9,r7,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b5,r5,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b1,r10,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b5,r5,band_UNPAIRED__condX2 <= 2
 0 <= X__b4,r5,band_LOW_PAIRED <= 2
 0 <= aux_cap_helper__b1,r5,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_b6,r12,band_LOW_PAIRED <= 536870910
 0 <= sv_function_helpervar_b5,r11__Z3 <= 1
 0 <= sv_function_helpervar_b5,r11__Z2 <= 1
 0 <= sv_function_helpervar_b5,r11__Z1 <= 1
 0 <= aux_cap_helper__b9,r7,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b9,r7,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_b4,r9,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b9,r4,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b9,r4,band_LOW_PAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b6,r1__condX3 <= 536870910
 0 <= sv_function_helpervar_b6,r1__condX2 <= 536870910
 0 <= aux_cap_b6,r3,band_LOW_PAIRED <= 536870910
 0 <= sv_function_helpervar_b6,r1__condX1 <= 536870910
 0 <= sv_function_helpervar_b4,r3__Z1 <= 1
 0 <= sv_function_helpervar_b4,r3__Z2 <= 1
 0 <= sv_function_helpervar_b4,r3__Z3 <= 1
 0 <= aux_cap_helper__b4,r13,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b8,r10,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b8,r10,band_UNPAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b3,r2__condX3 <= 536870910
 0 <= sv_function_helpervar_b3,r2__condX2 <= 536870910
 0 <= sv_function_helpervar_b3,r2__condX1 <= 536870910
 0 <= aux_cap_helper__b4,r13,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b8,r2,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_b9,r0,band_UNPAIRED <= 536870910
 0 <= X__b4,r13,band_HIGH_PAIRED <= 3
 0 <= aux_cap_helper__b6,r6,band_LOW_PAIRED__Z1 <= 1
 0 <= X__b8,r3,band_HIGH_PAIRED <= 3
 0 <= sv_function_helpervar_b3,r8__condX3 <= 536870910
 0 <= X__b8,r12,band_UNPAIRED <= 2
 0 <= X__b6,r9,band_HIGH_PAIRED <= 3
 0 <= sv_function_helpervar_b3,r8__condX2 <= 536870910
 0 <= aux_cap_helper__b8,r2,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b8,r2,band_HIGH_PAIRED__Z3 <= 1
 0 <= sv_function_helpervar_b3,r8__condX1 <= 536870910
 0 <= aux_cap_helper__b5,r5,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_b6,r9,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b2,r0,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b5,r5,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b5,r5,band_HIGH_PAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b1,r5__condX3 <= 536870910
 0 <= sv_function_helpervar_b7,r12__condX3 <= 536870910
 0 <= sv_function_helpervar_b7,r12__condX2 <= 536870910
 0 <= aux_cap_helper__b6,r6,band_LOW_PAIRED__Z2 <= 1
 0 <= sv_function_helpervar_b7,r12__condX1 <= 536870910
 0 <= sv_function_helpervar_b1,r5__condX2 <= 536870910
 0 <= sv_function_helpervar_b1,r5__condX1 <= 536870910
 0 <= aux_cap_helper__b3,r9,band_UNPAIRED__condX2 <= 2
 0 <= X__b8,r6,band_HIGH_PAIRED <= 3
 0 <= aux_cap_helper__b8,r9,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b8,r9,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b2,r0,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b3,r9,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b4,r12,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b4,r12,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b4,r12,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b6,r6,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b6,r6,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b6,r6,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b6,r5,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b6,r5,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b1,r10,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b1,r10,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b1,r10,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b6,r5,band_HIGH_PAIRED__Z3 <= 1
 0 <= X__b6,r6,band_UNPAIRED <= 2
 0 <= sv_function_helpervar_b5,r7__Z3 <= 1
 0 <= sv_function_helpervar_b5,r7__Z1 <= 1
 0 <= sv_function_helpervar_b5,r7__Z2 <= 1
 0 <= sv_function_helpervar_b8,r11__condX3 <= 536870910
 0 <= sv_function_helpervar_b8,r11__condX1 <= 536870910
 0 <= sv_function_helpervar_b8,r11__condX2 <= 536870910
 0 <= X__b0,r1,band_HIGH_PAIRED <= 3
 0 <= aux_cap_helper__b8,r2,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b8,r1,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b8,r1,band_UNPAIRED__condX2 <= 2
 0 <= X__b1,r3,band_HIGH_PAIRED <= 3
 0 <= aux_cap_helper__b1,r2,band_LOW_PAIRED__condX2 <= 2
 0 <= sv_function_helpervar_b8,r5__Z1 <= 1
 0 <= aux_cap_helper__b1,r2,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b8,r2,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_b3,r10,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_helper__b4,r10,band_HIGH_PAIRED__Z3 <= 1
 0 <= X__b5,r0,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b4,r10,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_b4,r5,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b4,r10,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b5,r1,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_b3,r7,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_helper__b5,r1,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b0,r5,band_LOW_PAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b8,r5__Z3 <= 1
 0 <= aux_cap_helper__b0,r5,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b7,r11,band_HIGH_PAIRED__Z3 <= 1
 0 <= sv_function_helpervar_b8,r5__Z2 <= 1
 0 <= sv_function_helpervar_b8,r2__condX3 <= 536870910
 0 <= sv_function_helpervar_b8,r2__condX2 <= 536870910
 0 <= sv_function_helpervar_b8,r2__condX1 <= 536870910
 0 <= X__b1,r8,band_HIGH_PAIRED <= 3
 0 <= sv_function_helpervar_b4,r3__condX1 <= 536870910
 0 <= sv_function_helpervar_b4,r3__condX3 <= 536870910
 0 <= X__b9,r5,band_UNPAIRED <= 2
 0 <= sv_function_helpervar_b4,r3__condX2 <= 536870910
 0 <= aux_cap_helper__b1,r6,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b1,r6,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b1,r6,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_b3,r11,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b8,r9,band_LOW_PAIRED__condX1 <= 2
 0 <= X__b1,r10,band_LOW_PAIRED <= 2
 0 <= aux_cap_helper__b8,r9,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_b2,r6,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b1,r3,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b1,r3,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b1,r3,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b1,r11,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b1,r11,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b1,r11,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_b6,r1,band_LOW_PAIRED <= 536870910
 0 <= X__b8,r0,band_LOW_PAIRED <= 2
 0 <= X__b5,r6,band_LOW_PAIRED <= 2
 0 <= X__b6,r12,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b0,r6,band_HIGH_PAIRED__condX1 <= 3
 0 <= sv_function_helpervar_b1,r1__Z3 <= 1
 0 <= aux_cap_helper__b0,r6,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b0,r6,band_HIGH_PAIRED__condX2 <= 3
 0 <= sv_function_helpervar_b1,r1__Z1 <= 1
 0 <= sv_function_helpervar_b1,r1__Z2 <= 1
 0 <= aux_cap_helper__b1,r9,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b1,r9,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b5,r8,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b8,r0,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b8,r0,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b5,r8,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b5,r8,band_HIGH_PAIRED__Z2 <= 1
 0 <= sv_function_helpervar_b0,r13__Z2 <= 1
 0 <= aux_cap_helper__b8,r12,band_UNPAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b0,r13__Z1 <= 1
 0 <= aux_cap_helper__b8,r12,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b0,r8,band_UNPAIRED__Z2 <= 1
 0 <= sv_function_helpervar_b0,r13__Z3 <= 1
 0 <= aux_cap_helper__b4,r13,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b0,r8,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b4,r13,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b9,r1,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b9,r1,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b7,r4,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b0,r11,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b7,r4,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b7,r4,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b3,r5,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b0,r11,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b3,r5,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_b7,r4,band_UNPAIRED <= 536870910
 0 <= sv_function_helpervar_b7,r1__condX2 <= 536870910
 0 <= sv_function_helpervar_b7,r1__condX1 <= 536870910
 0 <= aux_cap_b6,r12,band_UNPAIRED <= 536870910
 0 <= sv_function_helpervar_b7,r1__condX3 <= 536870910
 0 <= aux_cap_helper__b3,r1,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b3,r1,band_HIGH_PAIRED__condX3 <= 3
 0 <= X__b3,r12,band_UNPAIRED <= 2
 0 <= aux_sv_b1,r11 <= 536870910
 0 <= aux_sv_b1,r12 <= 536870910
 0 <= aux_cap_b2,r10,band_HIGH_PAIRED <= 536870910
 0 <= aux_sv_b1,r10 <= 536870910
 0 <= aux_cap_helper__b2,r7,band_HIGH_PAIRED__condX1 <= 3
 0 <= sv_function_helpervar_b4,r11__Z3 <= 1
 0 <= aux_cap_helper__b2,r7,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b2,r7,band_HIGH_PAIRED__condX3 <= 3
 0 <= sv_function_helpervar_b4,r11__Z1 <= 1
 0 <= sv_function_helpervar_b4,r11__Z2 <= 1
 0 <= X__b7,r4,band_UNPAIRED <= 2
 0 <= aux_cap_b3,r8,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b3,r1,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_sv_b1,r13 <= 536870910
 0 <= aux_cap_b4,r5,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_helper__b3,r7,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_b2,r11,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b3,r7,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b3,r7,band_HIGH_PAIRED__condX3 <= 3
 0 <= sv_function_helpervar_b6,r0__Z1 <= 1
 0 <= sv_function_helpervar_b6,r0__Z2 <= 1
 0 <= sv_function_helpervar_b6,r0__Z3 <= 1
 0 <= aux_cap_b9,r12,band_UNPAIRED <= 536870910
 0 <= aux_cap_b9,r8,band_UNPAIRED <= 536870910
 0 <= aux_cap_b8,r2,band_UNPAIRED <= 536870910
 0 <= aux_sv_b8,r6 <= 536870910
 0 <= aux_sv_b8,r5 <= 536870910
 0 <= aux_sv_b8,r8 <= 536870910
 0 <= aux_sv_b8,r7 <= 536870910
 0 <= sv_function_helpervar_b7,r6__condX3 <= 536870910
 0 <= sv_function_helpervar_b7,r6__condX2 <= 536870910
 0 <= aux_sv_b8,r9 <= 536870910
 0 <= sv_function_helpervar_b7,r6__condX1 <= 536870910
 0 <= aux_sv_b8,r0 <= 536870910
 0 <= aux_sv_b8,r2 <= 536870910
 0 <= aux_sv_b8,r1 <= 536870910
 0 <= aux_sv_b8,r4 <= 536870910
 0 <= aux_sv_b8,r3 <= 536870910
 0 <= X__b9,r10,band_LOW_PAIRED <= 2
 0 <= aux_cap_helper__b5,r3,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b5,r3,band_LOW_PAIRED__condX1 <= 2
 0 <= X__b3,r0,band_HIGH_PAIRED <= 3
 0 <= aux_cap_b9,r9,band_LOW_PAIRED <= 536870910
 0 <= X__b5,r10,band_HIGH_PAIRED <= 3
 0 <= aux_cap_b1,r9,band_LOW_PAIRED <= 536870910
 0 <= sv_function_helpervar_b6,r4__Z2 <= 1
 0 <= sv_function_helpervar_b6,r4__Z1 <= 1
 0 <= X__b5,r9,band_HIGH_PAIRED <= 3
 0 <= aux_cap_b1,r10,band_HIGH_PAIRED <= 536870910
 0 <= sv_function_helpervar_b6,r4__Z3 <= 1
 0 <= X__b2,r10,band_LOW_PAIRED <= 2
 0 <= aux_cap_helper__b6,r4,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_b5,r0,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b0,r8,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b0,r8,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b6,r4,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b8,r6,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b4,r0,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b6,r2,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b4,r0,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b6,r2,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_b4,r12,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b6,r2,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b8,r6,band_UNPAIRED__condX1 <= 2
 0 <= X__b2,r9,band_LOW_PAIRED <= 2
 0 <= aux_cap_b6,r7,band_HIGH_PAIRED <= 536870910
 0 <= X__b2,r13,band_HIGH_PAIRED <= 3
 0 <= aux_cap_b7,r12,band_UNPAIRED <= 536870910
 0 <= X__b6,r2,band_UNPAIRED <= 2
 0 <= aux_cap_b9,r11,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b4,r9,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b4,r9,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_b0,r11,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b8,r4,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b8,r4,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b8,r4,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b3,r12,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_Omega_b7,r10 <= 536870910
 0 <= aux_Omega_b7,r11 <= 536870910
 0 <= aux_cap_b5,r7,band_UNPAIRED <= 536870910
 0 <= aux_Omega_b7,r12 <= 536870910
 0 <= aux_Omega_b7,r13 <= 536870910
 0 <= aux_cap_helper__b3,r12,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_b3,r6,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_helper__b5,r7,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_b7,r0,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_helper__b5,r7,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b5,r7,band_HIGH_PAIRED__Z3 <= 1
 0 <= X__b3,r8,band_LOW_PAIRED <= 2
 0 <= aux_cap_helper__b4,r11,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b4,r11,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b4,r11,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b8,r3,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b8,r3,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b8,r3,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_c_b0,r10 <= 536870910
 0 <= aux_c_b0,r11 <= 536870910
 0 <= aux_c_b0,r12 <= 536870910
 0 <= aux_c_b0,r13 <= 536870910
 0 <= aux_cap_helper__b7,r0,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b7,r0,band_LOW_PAIRED__Z2 <= 1
 0 <= sv_function_helpervar_b3,r6__Z1 <= 1
 0 <= aux_cap_b9,r6,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b9,r9,band_UNPAIRED__condX1 <= 2
 0 <= sv_function_helpervar_b3,r6__Z3 <= 1
 0 <= aux_cap_helper__b9,r9,band_UNPAIRED__condX2 <= 2
 0 <= sv_function_helpervar_b3,r6__Z2 <= 1
 0 <= X__b9,r1,band_UNPAIRED <= 2
 0 <= X__b0,r11,band_HIGH_PAIRED <= 3
 0 <= sv_function_helpervar_b2,r12__Z3 <= 1
 0 <= sv_function_helpervar_b2,r12__Z2 <= 1
 0 <= aux_cap_b8,r6,band_UNPAIRED <= 536870910
 0 <= sv_function_helpervar_b2,r12__Z1 <= 1
 0 <= sv_function_helpervar_b8,r7__condX1 <= 536870910
 0 <= sv_function_helpervar_b8,r7__condX3 <= 536870910
 0 <= sv_function_helpervar_b8,r7__condX2 <= 536870910
 0 <= aux_cap_helper__b3,r11,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b3,r11,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_b7,r9,band_HIGH_PAIRED <= 536870910
 0 <= X__b8,r12,band_HIGH_PAIRED <= 3
 0 <= aux_cap_helper__b4,r13,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b4,r13,band_UNPAIRED__condX1 <= 2
 0 <= X__b4,r0,band_HIGH_PAIRED <= 3
 0 <= X__b8,r3,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b1,r0,band_LOW_PAIRED__Z2 <= 1
 0 <= sv_function_helpervar_b3,r11__Z2 <= 1
 0 <= aux_cap_helper__b1,r0,band_LOW_PAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b3,r11__Z3 <= 1
 0 <= aux_cap_helper__b8,r9,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b4,r8,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b8,r9,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b8,r9,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b4,r8,band_LOW_PAIRED__Z2 <= 1
 0 <= X__b4,r4,band_LOW_PAIRED <= 2
 0 <= aux_cap_helper__b0,r1,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b0,r1,band_HIGH_PAIRED__condX1 <= 3
 0 <= sv_function_helpervar_b7,r7__Z1 <= 1
 0 <= aux_cap_helper__b0,r1,band_HIGH_PAIRED__condX3 <= 3
 0 <= sv_function_helpervar_b7,r7__Z2 <= 1
 0 <= sv_function_helpervar_b7,r7__Z3 <= 1
 0 <= sv_function_helpervar_b3,r11__Z1 <= 1
 0 <= aux_cap_helper__b7,r2,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b7,r2,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b5,r7,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b5,r7,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_b1,r7,band_LOW_PAIRED <= 536870910
 0 <= X__b6,r11,band_LOW_PAIRED <= 2
 0 <= aux_cap_helper__b5,r7,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_b1,r1,band_LOW_PAIRED <= 536870910
 0 <= sv_function_helpervar_b2,r6__Z1 <= 1
 0 <= aux_cap_b8,r4,band_HIGH_PAIRED <= 536870910
 0 <= X__b9,r5,band_HIGH_PAIRED <= 3
 0 <= sv_function_helpervar_b2,r6__Z2 <= 1
 0 <= sv_function_helpervar_b2,r6__Z3 <= 1
 0 <= sv_function_helpervar_b8,r2__Z3 <= 1
 0 <= X__b1,r1,band_UNPAIRED <= 2
 0 <= sv_function_helpervar_b8,r2__Z2 <= 1
 0 <= sv_function_helpervar_b8,r2__Z1 <= 1
 0 <= X__b4,r12,band_HIGH_PAIRED <= 3
 0 <= aux_cap_helper__b3,r13,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b3,r13,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b5,r7,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b3,r13,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b5,r7,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b8,r5,band_HIGH_PAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b9,r5__condX1 <= 536870910
 0 <= aux_cap_helper__b8,r5,band_HIGH_PAIRED__Z2 <= 1
 0 <= sv_function_helpervar_b9,r5__condX2 <= 536870910
 0 <= aux_cap_helper__b8,r5,band_HIGH_PAIRED__Z3 <= 1
 0 <= X__b6,r9,band_LOW_PAIRED <= 2
 0 <= sv_function_helpervar_b9,r5__condX3 <= 536870910
 0 <= aux_cap_helper__b6,r2,band_LOW_PAIRED__Z2 <= 1
 0 <= X__b1,r0,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b0,r7,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b6,r2,band_LOW_PAIRED__Z1 <= 1
 0 <= X__b1,r2,band_HIGH_PAIRED <= 3
 0 <= X__b0,r6,band_LOW_PAIRED <= 2
 0 <= aux_cap_helper__b0,r7,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_c_b8,r0 <= 536870910
 0 <= aux_c_b8,r1 <= 536870910
 0 <= aux_c_b8,r2 <= 536870910
 0 <= aux_cap_helper__b3,r3,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_c_b6,r11 <= 536870910
 0 <= aux_c_b6,r10 <= 536870910
 0 <= aux_c_b6,r13 <= 536870910
 0 <= aux_cap_helper__b4,r3,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_c_b6,r12 <= 536870910
 0 <= aux_cap_b3,r12,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_b4,r1,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b1,r5,band_HIGH_PAIRED__condX2 <= 3
 0 <= sv_function_helpervar_b1,r13__condX3 <= 536870910
 0 <= aux_cap_helper__b1,r5,band_HIGH_PAIRED__condX3 <= 3
 0 <= sv_function_helpervar_b1,r13__condX2 <= 536870910
 0 <= GlobalBidder_PSI__i8,k2 <= 536870910
 0 <= aux_cap_b9,r1,band_HIGH_PAIRED <= 536870910
 0 <= GlobalBidder_PSI__i8,k1 <= 536870910
 0 <= aux_cap_helper__b1,r5,band_HIGH_PAIRED__condX1 <= 3
 0 <= GlobalBidder_PSI__i8,k0 <= 536870910
 0 <= sv_function_helpervar_b1,r13__condX1 <= 536870910
 0 <= aux_cap_helper__b4,r3,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b0,r13,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b4,r3,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b0,r13,band_LOW_PAIRED__Z2 <= 1
 0 <= sv_function_helpervar_b3,r13__condX3 <= 536870910
 0 <= sv_function_helpervar_b3,r13__condX2 <= 536870910
 0 <= X__b2,r6,band_HIGH_PAIRED <= 3
 0 <= sv_function_helpervar_b3,r13__condX1 <= 536870910
 0 <= sv_function_helpervar_b9,r13__condX3 <= 536870910
 0 <= sv_function_helpervar_b2,r2__condX3 <= 536870910
 0 <= aux_cap_helper__b8,r11,band_UNPAIRED__condX2 <= 2
 0 <= sv_function_helpervar_b9,r13__condX2 <= 536870910
 0 <= aux_cap_helper__b8,r11,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_b2,r2,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b5,r13,band_HIGH_PAIRED__Z3 <= 1
 0 <= sv_function_helpervar_b9,r13__condX1 <= 536870910
 0 <= sv_function_helpervar_b7,r13__condX3 <= 536870910
 0 <= sv_function_helpervar_b7,r13__condX2 <= 536870910
 0 <= sv_function_helpervar_b2,r2__condX2 <= 536870910
 0 <= sv_function_helpervar_b7,r13__condX1 <= 536870910
 0 <= sv_function_helpervar_b2,r2__condX1 <= 536870910
 0 <= X__b1,r2,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b8,r3,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b8,r3,band_LOW_PAIRED__condX2 <= 2
 0 <= X__b0,r12,band_HIGH_PAIRED <= 3
 0 <= aux_cap_helper__b2,r9,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b2,r9,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b6,r8,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b2,r9,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b6,r8,band_LOW_PAIRED__condX2 <= 2
 0 <= X__b9,r9,band_UNPAIRED <= 2
 0 <= aux_sv_b5,r13 <= 536870910
 0 <= aux_sv_b5,r11 <= 536870910
 0 <= aux_sv_b5,r12 <= 536870910
 0 <= sv_function_helpervar_b6,r0__condX3 <= 536870910
 0 <= sv_function_helpervar_b6,r0__condX2 <= 536870910
 0 <= aux_sv_b5,r10 <= 536870910
 0 <= sv_function_helpervar_b6,r0__condX1 <= 536870910
 0 <= aux_cap_b4,r2,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b2,r2,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b2,r2,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_b6,r13,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_helper__b3,r3,band_LOW_PAIRED__Z2 <= 1
 0 <= sv_function_helpervar_b2,r5__condX2 <= 536870910
 0 <= sv_function_helpervar_b2,r5__condX3 <= 536870910
 0 <= aux_cap_helper__b3,r8,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b3,r8,band_HIGH_PAIRED__Z2 <= 1
 0 <= sv_function_helpervar_b2,r5__condX1 <= 536870910
 0 <= aux_cap_helper__b3,r8,band_HIGH_PAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b0,r3__condX2 <= 536870910
 0 <= aux_cap_helper__b3,r0,band_HIGH_PAIRED__Z2 <= 1
 0 <= sv_function_helpervar_b0,r3__condX3 <= 536870910
 0 <= aux_cap_helper__b3,r0,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_b7,r11,band_LOW_PAIRED <= 536870910
 0 <= X__b7,r3,band_LOW_PAIRED <= 2
 0 <= sv_function_helpervar_b0,r3__condX1 <= 536870910
 0 <= aux_cap_helper__b3,r0,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b9,r11,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b9,r11,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b9,r11,band_HIGH_PAIRED__condX3 <= 3
 0 <= X__b9,r10,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b8,r1,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b8,r2,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b8,r1,band_LOW_PAIRED__condX1 <= 2
 0 <= X__b5,r4,band_LOW_PAIRED <= 2
 0 <= aux_cap_helper__b2,r13,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b2,r13,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b8,r2,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b5,r4,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b5,r4,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_b4,r6,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_helper__b0,r12,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b0,r12,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_b2,r8,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b2,r13,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b2,r10,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b2,r10,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b2,r13,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b2,r10,band_HIGH_PAIRED__condX1 <= 3
 0 <= sv_function_helpervar_b9,r13__Z2 <= 1
 0 <= aux_cap_helper__b6,r11,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b8,r4,band_UNPAIRED__condX2 <= 2
 0 <= sv_function_helpervar_b9,r13__Z3 <= 1
 0 <= aux_cap_helper__b8,r4,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b6,r11,band_UNPAIRED__condX1 <= 2
 0 <= sv_function_helpervar_b6,r6__condX2 <= 536870910
 0 <= sv_function_helpervar_b6,r6__condX1 <= 536870910
 0 <= sv_function_helpervar_b6,r6__condX3 <= 536870910
 0 <= aux_cap_helper__b7,r1,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b2,r11,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b7,r1,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b2,r11,band_LOW_PAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b9,r13__Z1 <= 1
 0 <= sv_function_helpervar_b5,r10__condX1 <= 536870910
 0 <= sv_function_helpervar_b5,r10__condX2 <= 536870910
 0 <= X__b4,r6,band_HIGH_PAIRED <= 3
 0 <= sv_function_helpervar_b5,r10__condX3 <= 536870910
 0 <= X__b7,r0,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b5,r13,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b8,r3,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b5,r13,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b8,r3,band_LOW_PAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b5,r6__Z1 <= 1
 0 <= sv_function_helpervar_b5,r6__Z2 <= 1
 0 <= sv_function_helpervar_b5,r6__Z3 <= 1
 0 <= aux_cap_helper__b1,r7,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_b7,r1,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b9,r6,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b9,r6,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_b0,r8,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b1,r7,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b8,r5,band_LOW_PAIRED__Z2 <= 1
 0 <= sv_function_helpervar_b9,r10__condX1 <= 536870910
 0 <= aux_cap_helper__b8,r5,band_LOW_PAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b9,r10__condX2 <= 536870910
 0 <= sv_function_helpervar_b9,r10__condX3 <= 536870910
 0 <= aux_cap_helper__b3,r4,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b3,r4,band_HIGH_PAIRED__Z2 <= 1
 0 <= X__b9,r11,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b3,r4,band_HIGH_PAIRED__Z3 <= 1
 0 <= X__b7,r4,band_HIGH_PAIRED <= 3
 0 <= X__b7,r9,band_LOW_PAIRED <= 2
 0 <= X__b7,r10,band_HIGH_PAIRED <= 3
 0 <= aux_cap_b7,r4,band_LOW_PAIRED <= 536870910
 0 <= X__b1,r6,band_LOW_PAIRED <= 2
 0 <= X__b1,r0,band_LOW_PAIRED <= 2
 0 <= sv_function_helpervar_b3,r1__Z3 <= 1
 0 <= aux_cap_helper__b7,r3,band_HIGH_PAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b3,r1__Z2 <= 1
 0 <= aux_cap_helper__b7,r3,band_HIGH_PAIRED__Z2 <= 1
 0 <= sv_function_helpervar_b3,r1__Z1 <= 1
 0 <= aux_cap_helper__b7,r3,band_HIGH_PAIRED__Z3 <= 1
 0 <= sv_function_helpervar_b4,r4__condX2 <= 536870910
 0 <= sv_function_helpervar_b4,r4__condX1 <= 536870910
 0 <= sv_function_helpervar_b4,r4__condX3 <= 536870910
 0 <= aux_cap_b4,r12,band_LOW_PAIRED <= 536870910
 0 <= X__b8,r4,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b8,r12,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b8,r12,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_b3,r10,band_UNPAIRED <= 536870910
 0 <= X__b2,r5,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b9,r12,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b9,r12,band_HIGH_PAIRED__condX2 <= 3
 0 <= X__b1,r10,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b9,r12,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b3,r10,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b3,r10,band_LOW_PAIRED__condX1 <= 2
 0 <= sv_function_helpervar_b7,r3__condX3 <= 536870910
 0 <= X__b8,r5,band_UNPAIRED <= 2
 0 <= sv_function_helpervar_b7,r3__condX1 <= 536870910
 0 <= GlobalBidder_W_ir__b9,r11 <= 1
 0 <= sv_function_helpervar_b7,r3__condX2 <= 536870910
 0 <= GlobalBidder_W_ir__b9,r12 <= 1
 0 <= aux_cap_helper__b1,r9,band_LOW_PAIRED__Z2 <= 1
 0 <= GlobalBidder_W_ir__b9,r10 <= 1
 0 <= aux_cap_helper__b1,r9,band_LOW_PAIRED__Z1 <= 1
 0 <= GlobalBidder_W_ir__b9,r13 <= 1
 0 <= aux_cap_helper__b8,r3,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b8,r3,band_UNPAIRED__condX1 <= 2
 0 <= aux_sv_b2,r5 <= 536870910
 0 <= aux_cap_helper__b8,r5,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_sv_b2,r6 <= 536870910
 0 <= aux_cap_helper__b8,r5,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_sv_b2,r3 <= 536870910
 0 <= aux_sv_b2,r4 <= 536870910
 0 <= aux_cap_helper__b8,r5,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_sv_b2,r1 <= 536870910
 0 <= X__b2,r6,band_UNPAIRED <= 2
 0 <= aux_sv_b2,r2 <= 536870910
 0 <= aux_sv_b2,r0 <= 536870910
 0 <= X__b8,r6,band_UNPAIRED <= 2
 0 <= aux_sv_b2,r9 <= 536870910
 0 <= X__b1,r11,band_UNPAIRED <= 2
 0 <= X__b2,r8,band_UNPAIRED <= 2
 0 <= aux_sv_b2,r7 <= 536870910
 0 <= aux_cap_helper__b3,r5,band_UNPAIRED__Z1 <= 1
 0 <= aux_sv_b2,r8 <= 536870910
 0 <= aux_Omega_b3,r11 <= 536870910
 0 <= aux_Omega_b3,r12 <= 536870910
 0 <= aux_cap_helper__b3,r1,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_Omega_b3,r13 <= 536870910
 0 <= aux_cap_helper__b3,r1,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_b2,r12,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_b9,r11,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_helper__b9,r3,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b9,r5,band_UNPAIRED__condX2 <= 2
 0 <= aux_Omega_b3,r10 <= 536870910
 0 <= aux_cap_helper__b9,r5,band_UNPAIRED__condX1 <= 2
 0 <= X__b2,r7,band_UNPAIRED <= 2
 0 <= sv_function_helpervar_b0,r11__Z1 <= 1
 0 <= sv_function_helpervar_b0,r11__Z2 <= 1
 0 <= aux_cap_helper__b3,r5,band_UNPAIRED__Z2 <= 1
 0 <= sv_function_helpervar_b0,r11__Z3 <= 1
 0 <= aux_cap_helper__b9,r3,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_b0,r12,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_helper__b6,r4,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b6,r4,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_b2,r12,band_UNPAIRED <= 536870910
 0 <= X__b4,r11,band_LOW_PAIRED <= 2
 0 <= sv_function_helpervar_b6,r11__condX3 <= 536870910
 0 <= aux_sv_b9,r13 <= 536870910
 0 <= sv_function_helpervar_b2,r2__Z3 <= 1
 0 <= aux_cap_helper__b6,r0,band_HIGH_PAIRED__condX1 <= 3
 0 <= sv_function_helpervar_b5,r1__condX2 <= 536870910
 0 <= sv_function_helpervar_b6,r11__condX1 <= 536870910
 0 <= sv_function_helpervar_b5,r1__condX1 <= 536870910
 0 <= sv_function_helpervar_b6,r11__condX2 <= 536870910
 0 <= aux_cap_helper__b0,r11,band_UNPAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b5,r1__condX3 <= 536870910
 0 <= aux_cap_helper__b0,r5,band_LOW_PAIRED__condX1 <= 2
 0 <= sv_function_helpervar_b2,r2__Z1 <= 1
 0 <= aux_cap_helper__b6,r0,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b0,r5,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b0,r11,band_UNPAIRED__Z2 <= 1
 0 <= sv_function_helpervar_b2,r2__Z2 <= 1
 0 <= aux_cap_helper__b6,r0,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_b5,r6,band_LOW_PAIRED <= 536870910
 0 <= sv_function_helpervar_b8,r8__condX3 <= 536870910
 0 <= X__b9,r7,band_LOW_PAIRED <= 2
 0 <= aux_sv_b9,r12 <= 536870910
 0 <= aux_sv_b9,r11 <= 536870910
 0 <= sv_function_helpervar_b8,r8__condX2 <= 536870910
 0 <= aux_sv_b9,r10 <= 536870910
 0 <= sv_function_helpervar_b8,r8__condX1 <= 536870910
 0 <= aux_cap_helper__b9,r7,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b9,r7,band_UNPAIRED__Z2 <= 1
 0 <= X__b8,r13,band_UNPAIRED <= 2
 0 <= X__b3,r2,band_HIGH_PAIRED <= 3
 0 <= X__b9,r12,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b3,r9,band_UNPAIRED__Z1 <= 1
 0 <= X__b8,r3,band_LOW_PAIRED <= 2
 0 <= sv_function_helpervar_b0,r5__Z1 <= 1
 0 <= aux_cap_helper__b3,r9,band_UNPAIRED__Z2 <= 1
 0 <= X__b4,r13,band_UNPAIRED <= 2
 0 <= sv_function_helpervar_b0,r5__Z2 <= 1
 0 <= sv_function_helpervar_b0,r5__Z3 <= 1
 0 <= sv_function_helpervar_b7,r3__Z1 <= 1
 0 <= sv_function_helpervar_b7,r3__Z2 <= 1
 0 <= aux_cap_helper__b1,r8,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b1,r8,band_HIGH_PAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b8,r10__Z1 <= 1
 0 <= X__b6,r0,band_HIGH_PAIRED <= 3
 0 <= aux_cap_helper__b7,r0,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b7,r0,band_HIGH_PAIRED__Z2 <= 1
 0 <= sv_function_helpervar_b8,r10__Z2 <= 1
 0 <= aux_cap_helper__b1,r8,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b7,r0,band_HIGH_PAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b8,r10__Z3 <= 1
 0 <= X__b9,r4,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b6,r6,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_b0,r2,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b6,r6,band_LOW_PAIRED__condX1 <= 2
 0 <= sv_function_helpervar_b7,r3__Z3 <= 1
 0 <= X__b5,r9,band_LOW_PAIRED <= 2
 0 <= aux_cap_helper__b7,r5,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b7,r5,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b8,r7,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_b6,r4,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_b6,r10,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b8,r7,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b2,r9,band_UNPAIRED__Z1 <= 1
 0 <= X__b8,r9,band_LOW_PAIRED <= 2
 0 <= aux_cap_helper__b2,r9,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b9,r13,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b9,r13,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b7,r10,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b7,r10,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b7,r1,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_b4,r2,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b7,r9,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b7,r9,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b1,r1,band_HIGH_PAIRED__Z2 <= 1
 0 <= X__b5,r13,band_LOW_PAIRED <= 2
 0 <= aux_cap_helper__b1,r1,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b1,r1,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b1,r8,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_b5,r0,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b1,r8,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b7,r1,band_UNPAIRED__Z2 <= 1
 0 <= X__b3,r4,band_LOW_PAIRED <= 2
 0 <= aux_cap_helper__b4,r12,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b4,r12,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b6,r0,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b6,r0,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b6,r10,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b6,r10,band_UNPAIRED__condX1 <= 2
 0 <= X__b2,r0,band_UNPAIRED <= 2
 0 <= X__b1,r7,band_UNPAIRED <= 2
 0 <= aux_cap_b8,r13,band_LOW_PAIRED <= 536870910
 0 <= X__b4,r11,band_HIGH_PAIRED <= 3
 0 <= aux_cap_b6,r6,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b5,r5,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b8,r5,band_UNPAIRED__condX2 <= 2
 0 <= X__b9,r1,band_LOW_PAIRED <= 2
 0 <= aux_cap_helper__b5,r5,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b8,r5,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b2,r5,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b2,r5,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b8,r3,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b8,r3,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_b4,r4,band_UNPAIRED <= 536870910
 0 <= aux_cap_b6,r13,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b5,r0,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b5,r0,band_LOW_PAIRED__condX2 <= 2
 0 <= sv_function_helpervar_b1,r9__condX3 <= 536870910
 0 <= sv_function_helpervar_b3,r1__condX3 <= 536870910
 0 <= aux_sv_b5,r2 <= 536870910
 0 <= aux_sv_b5,r3 <= 536870910
 0 <= sv_function_helpervar_b3,r1__condX1 <= 536870910
 0 <= aux_sv_b5,r4 <= 536870910
 0 <= sv_function_helpervar_b3,r1__condX2 <= 536870910
 0 <= aux_sv_b5,r5 <= 536870910
 0 <= aux_sv_b5,r6 <= 536870910
 0 <= aux_sv_b5,r7 <= 536870910
 0 <= aux_sv_b5,r8 <= 536870910
 0 <= aux_cap_b6,r8,band_UNPAIRED <= 536870910
 0 <= aux_sv_b5,r9 <= 536870910
 0 <= sv_function_helpervar_b1,r9__condX1 <= 536870910
 0 <= sv_function_helpervar_b1,r9__condX2 <= 536870910
 0 <= aux_sv_b5,r0 <= 536870910
 0 <= aux_sv_b5,r1 <= 536870910
 0 <= X__b5,r11,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b2,r0,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b2,r0,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b1,r3,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b5,r4,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b5,r4,band_HIGH_PAIRED__Z2 <= 1
 0 <= X__b2,r7,band_HIGH_PAIRED <= 3
 0 <= aux_cap_helper__b5,r4,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b8,r12,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b8,r12,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b5,r3,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b5,r3,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b8,r12,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b5,r3,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b1,r3,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b4,r1,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b4,r1,band_UNPAIRED__condX1 <= 2
 0 <= sv_function_helpervar_b5,r2__Z3 <= 1
 0 <= X__b0,r10,band_LOW_PAIRED <= 2
 0 <= sv_function_helpervar_b5,r2__Z2 <= 1
 0 <= sv_function_helpervar_b5,r2__Z1 <= 1
 0 <= aux_cap_helper__b2,r9,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b2,r6,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_b9,r9,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b2,r9,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b2,r6,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b2,r6,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b9,r9,band_HIGH_PAIRED__condX2 <= 3
 0 <= X__b7,r5,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b9,r9,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b0,r3,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_b7,r0,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b0,r3,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b9,r9,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b5,r10,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b5,r10,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b5,r10,band_HIGH_PAIRED__Z2 <= 1
 0 <= X__b9,r2,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b5,r8,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_b5,r8,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b5,r8,band_LOW_PAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b6,r13__Z3 <= 1
 0 <= sv_function_helpervar_b6,r13__Z2 <= 1
 0 <= sv_function_helpervar_b6,r13__Z1 <= 1
 0 <= aux_cap_b3,r6,band_HIGH_PAIRED <= 536870910
 0 <= sv_function_helpervar_b3,r10__condX3 <= 536870910
 0 <= aux_cap_helper__b2,r10,band_HIGH_PAIRED__Z2 <= 1
 0 <= sv_function_helpervar_b3,r10__condX1 <= 536870910
 0 <= aux_cap_helper__b2,r10,band_HIGH_PAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b3,r10__condX2 <= 536870910
 0 <= aux_cap_helper__b2,r10,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b1,r11,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b1,r11,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b0,r10,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_Omega_b7,r0 <= 536870910
 0 <= aux_cap_helper__b0,r10,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_Omega_b7,r1 <= 536870910
 0 <= aux_Omega_b7,r2 <= 536870910
 0 <= aux_cap_helper__b9,r8,band_UNPAIRED__condX2 <= 2
 0 <= aux_Omega_b7,r3 <= 536870910
 0 <= aux_cap_helper__b9,r8,band_UNPAIRED__condX1 <= 2
 0 <= X__b3,r6,band_UNPAIRED <= 2
 0 <= sv_function_helpervar_b4,r7__Z1 <= 1
 0 <= aux_cap_helper__b1,r0,band_HIGH_PAIRED__Z3 <= 1
 0 <= sv_function_helpervar_b4,r7__Z3 <= 1
 0 <= sv_function_helpervar_b4,r7__Z2 <= 1
 0 <= aux_cap_helper__b1,r0,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b1,r0,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b0,r4,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b0,r4,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b3,r11,band_HIGH_PAIRED__condX3 <= 3
 0 <= X__b1,r9,band_UNPAIRED <= 2
 0 <= X__b4,r9,band_LOW_PAIRED <= 2
 0 <= X__b2,r0,band_LOW_PAIRED <= 2
 0 <= aux_cap_helper__b3,r11,band_HIGH_PAIRED__condX2 <= 3
 0 <= X__b2,r6,band_LOW_PAIRED <= 2
 0 <= aux_cap_helper__b3,r11,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_c_b8,r3 <= 536870910
 0 <= aux_c_b8,r4 <= 536870910
 0 <= X__b3,r11,band_HIGH_PAIRED <= 3
 0 <= aux_cap_helper__b3,r5,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_c_b8,r5 <= 536870910
 0 <= aux_cap_helper__b3,r5,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_c_b8,r6 <= 536870910
 0 <= sv_function_helpervar_b6,r9__condX3 <= 536870910
 0 <= aux_c_b8,r7 <= 536870910
 0 <= aux_cap_helper__b8,r11,band_LOW_PAIRED__condX1 <= 2
 0 <= sv_function_helpervar_b6,r9__condX2 <= 536870910
 0 <= aux_c_b8,r8 <= 536870910
 0 <= sv_function_helpervar_b6,r9__condX1 <= 536870910
 0 <= aux_c_b8,r9 <= 536870910
 0 <= sv_function_helpervar_b0,r1__Z1 <= 1
 0 <= aux_cap_helper__b2,r5,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b2,r5,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_b0,r7,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_b0,r1,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_helper__b3,r5,band_HIGH_PAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b0,r1__Z3 <= 1
 0 <= X__b9,r0,band_HIGH_PAIRED <= 3
 0 <= sv_function_helpervar_b0,r1__Z2 <= 1
 0 <= X__b3,r4,band_UNPAIRED <= 2
 0 <= X__b8,r11,band_LOW_PAIRED <= 2
 0 <= aux_cap_b0,r7,band_HIGH_PAIRED <= 536870910
 0 <= X__b0,r3,band_UNPAIRED <= 2
 0 <= aux_cap_b5,r2,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b8,r11,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b8,r5,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b8,r5,band_LOW_PAIRED__condX1 <= 2
 0 <= X__b6,r1,band_HIGH_PAIRED <= 3
 0 <= aux_cap_helper__b2,r12,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_Omega_b7,r4 <= 536870910
 0 <= aux_cap_helper__b2,r12,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_Omega_b7,r5 <= 536870910
 0 <= aux_cap_b5,r10,band_HIGH_PAIRED <= 536870910
 0 <= aux_Omega_b7,r6 <= 536870910
 0 <= aux_Omega_b7,r7 <= 536870910
 0 <= sv_function_helpervar_b5,r3__condX3 <= 536870910
 0 <= aux_Omega_b7,r8 <= 536870910
 0 <= aux_Omega_b7,r9 <= 536870910
 0 <= sv_function_helpervar_b5,r3__condX1 <= 536870910
 0 <= sv_function_helpervar_b5,r3__condX2 <= 536870910
 0 <= sv_function_helpervar_b4,r7__condX2 <= 536870910
 0 <= sv_function_helpervar_b4,r7__condX3 <= 536870910
 0 <= sv_function_helpervar_b4,r7__condX1 <= 536870910
 0 <= X__b1,r12,band_LOW_PAIRED <= 2
 0 <= aux_cap_helper__b2,r12,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b6,r9,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b6,r9,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b6,r1,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b6,r1,band_UNPAIRED__Z2 <= 1
 0 <= sv_function_helpervar_b9,r8__Z1 <= 1
 0 <= aux_cap_helper__b6,r9,band_HIGH_PAIRED__Z3 <= 1
 0 <= sv_function_helpervar_b9,r8__Z2 <= 1
 0 <= sv_function_helpervar_b9,r8__Z3 <= 1
 0 <= aux_cap_helper__b9,r3,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b9,r3,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b0,r7,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b0,r7,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_b6,r11,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b0,r0,band_LOW_PAIRED__Z2 <= 1
 0 <= X__b0,r1,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b5,r2,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b5,r2,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_b7,r6,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b4,r11,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b4,r11,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_b7,r11,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b0,r0,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b8,r10,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b8,r10,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b8,r10,band_HIGH_PAIRED__condX1 <= 3
 0 <= sv_function_helpervar_b6,r7__condX1 <= 536870910
 0 <= X__b7,r7,band_UNPAIRED <= 2
 0 <= sv_function_helpervar_b6,r7__condX2 <= 536870910
 0 <= sv_function_helpervar_b6,r7__condX3 <= 536870910
 0 <= X__b1,r4,band_LOW_PAIRED <= 2
 0 <= aux_cap_helper__b0,r4,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b0,r4,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b0,r4,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b0,r5,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b0,r5,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b2,r6,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b8,r11,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b8,r11,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_b4,r1,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b2,r6,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b5,r4,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b5,r4,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b5,r4,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b3,r8,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b3,r8,band_LOW_PAIRED__condX2 <= 2
 0 <= X__b3,r12,band_HIGH_PAIRED <= 3
 0 <= aux_cap_b0,r3,band_LOW_PAIRED <= 536870910
 0 <= sv_function_helpervar_b7,r0__Z3 <= 1
 0 <= sv_function_helpervar_b7,r0__Z2 <= 1
 0 <= sv_function_helpervar_b7,r0__Z1 <= 1
 0 <= aux_cap_helper__b1,r8,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b1,r8,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b1,r8,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b2,r13,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b9,r9,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b2,r13,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b9,r9,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_b3,r4,band_UNPAIRED <= 536870910
 0 <= sv_function_helpervar_b0,r9__Z2 <= 1
 0 <= sv_function_helpervar_b1,r4__Z3 <= 1
 0 <= sv_function_helpervar_b0,r9__Z3 <= 1
 0 <= sv_function_helpervar_b1,r4__Z2 <= 1
 0 <= aux_cap_b4,r1,band_LOW_PAIRED <= 536870910
 0 <= sv_function_helpervar_b1,r4__Z1 <= 1
 0 <= aux_cap_helper__b6,r13,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b6,r13,band_LOW_PAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b0,r9__Z1 <= 1
 0 <= aux_cap_b1,r10,band_LOW_PAIRED <= 536870910
 0 <= X__b3,r0,band_LOW_PAIRED <= 2
 0 <= aux_cap_b9,r13,band_LOW_PAIRED <= 536870910
 0 <= sv_function_helpervar_b8,r13__Z1 <= 1
 0 <= sv_function_helpervar_b8,r13__Z2 <= 1
 0 <= X__b9,r3,band_LOW_PAIRED <= 2
 0 <= sv_function_helpervar_b8,r13__Z3 <= 1
 0 <= aux_cap_b3,r5,band_UNPAIRED <= 536870910
 0 <= aux_cap_b9,r2,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_b0,r3,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b5,r3,band_UNPAIRED__condX2 <= 2
 0 <= X__b7,r6,band_UNPAIRED <= 2
 0 <= X__b7,r7,band_LOW_PAIRED <= 2
 0 <= aux_cap_helper__b5,r4,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b5,r4,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b3,r3,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b3,r3,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b3,r3,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_b1,r3,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b5,r3,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b0,r1,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b0,r1,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_b8,r2,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_helper__b6,r13,band_LOW_PAIRED__condX1 <= 2
 0 <= sv_function_helpervar_b6,r5__Z3 <= 1
 0 <= sv_function_helpervar_b6,r5__Z2 <= 1
 0 <= sv_function_helpervar_b6,r5__Z1 <= 1
 0 <= sv_function_helpervar_b8,r0__condX3 <= 536870910
 0 <= sv_function_helpervar_b0,r12__condX2 <= 536870910
 0 <= sv_function_helpervar_b0,r2__condX3 <= 536870910
 0 <= sv_function_helpervar_b0,r12__condX3 <= 536870910
 0 <= sv_function_helpervar_b0,r2__condX1 <= 536870910
 0 <= sv_function_helpervar_b9,r5__Z3 <= 1
 0 <= X__b8,r13,band_LOW_PAIRED <= 2
 0 <= sv_function_helpervar_b0,r12__condX1 <= 536870910
 0 <= sv_function_helpervar_b0,r2__condX2 <= 536870910
 0 <= sv_function_helpervar_b4,r5__condX2 <= 536870910
 0 <= sv_function_helpervar_b4,r5__condX3 <= 536870910
 0 <= aux_cap_helper__b6,r10,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b6,r10,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b6,r13,band_LOW_PAIRED__condX2 <= 2
 0 <= sv_function_helpervar_b4,r5__condX1 <= 536870910
 0 <= aux_cap_helper__b8,r8,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b8,r8,band_HIGH_PAIRED__Z3 <= 1
 0 <= sv_function_helpervar_b6,r12__condX1 <= 536870910
 0 <= aux_cap_helper__b8,r8,band_HIGH_PAIRED__Z2 <= 1
 0 <= sv_function_helpervar_b6,r12__condX2 <= 536870910
 0 <= X__b7,r13,band_LOW_PAIRED <= 2
 0 <= sv_function_helpervar_b6,r12__condX3 <= 536870910
 0 <= sv_function_helpervar_b9,r5__Z1 <= 1
 0 <= sv_function_helpervar_b9,r5__Z2 <= 1
 0 <= X__b5,r5,band_HIGH_PAIRED <= 3
 0 <= sv_function_helpervar_b8,r0__condX1 <= 536870910
 0 <= sv_function_helpervar_b8,r0__condX2 <= 536870910
 0 <= aux_cap_helper__b3,r1,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_b5,r11,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b3,r1,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b3,r1,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b3,r2,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b2,r11,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b3,r2,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_b9,r3,band_UNPAIRED <= 536870910
 0 <= sv_function_helpervar_b3,r9__Z1 <= 1
 0 <= sv_function_helpervar_b3,r9__Z2 <= 1
 0 <= sv_function_helpervar_b3,r9__Z3 <= 1
 0 <= aux_cap_helper__b8,r12,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b8,r12,band_LOW_PAIRED__Z2 <= 1
 0 <= X__b8,r11,band_HIGH_PAIRED <= 3
 0 <= aux_cap_helper__b2,r11,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b4,r1,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b4,r1,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b3,r2,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_b5,r1,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_b7,r5,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b1,r13,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b1,r13,band_LOW_PAIRED__condX2 <= 2
 0 <= sv_function_helpervar_b4,r4__Z1 <= 1
 0 <= sv_function_helpervar_b4,r4__Z2 <= 1
 0 <= sv_function_helpervar_b4,r4__Z3 <= 1
 0 <= X__b9,r4,band_HIGH_PAIRED <= 3
 0 <= aux_cap_helper__b5,r12,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b5,r12,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_b1,r11,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b0,r10,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b0,r1,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b0,r10,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b0,r1,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b6,r6,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b6,r6,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_b7,r5,band_UNPAIRED <= 536870910
 0 <= aux_Omega_b1,r7 <= 536870910
 0 <= aux_Omega_b1,r6 <= 536870910
 0 <= aux_cap_helper__b0,r10,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_Omega_b1,r9 <= 536870910
 0 <= aux_Omega_b1,r8 <= 536870910
 0 <= aux_Omega_b1,r3 <= 536870910
 0 <= aux_Omega_b1,r2 <= 536870910
 0 <= aux_cap_helper__b5,r1,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_Omega_b1,r5 <= 536870910
 0 <= aux_cap_helper__b5,r1,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_Omega_b1,r4 <= 536870910
 0 <= aux_cap_helper__b5,r1,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_Omega_b1,r1 <= 536870910
 0 <= aux_cap_helper__b8,r10,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_Omega_b1,r0 <= 536870910
 0 <= aux_cap_helper__b8,r10,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b1,r2,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b1,r2,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b1,r2,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_b7,r6,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b8,r13,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b2,r11,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b2,r11,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b2,r11,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b4,r1,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b8,r13,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b4,r1,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b8,r13,band_HIGH_PAIRED__Z3 <= 1
 0 <= sv_function_helpervar_b8,r9__condX1 <= 536870910
 0 <= aux_cap_helper__b8,r13,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b8,r13,band_LOW_PAIRED__condX1 <= 2
 0 <= sv_function_helpervar_b8,r9__condX2 <= 536870910
 0 <= sv_function_helpervar_b8,r9__condX3 <= 536870910
 0 <= aux_cap_b1,r8,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b5,r12,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b6,r11,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b6,r11,band_LOW_PAIRED__Z2 <= 1
 0 <= X__b0,r8,band_LOW_PAIRED <= 2
 0 <= aux_cap_helper__b6,r4,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b6,r4,band_LOW_PAIRED__condX2 <= 2
 0 <= X__b4,r2,band_LOW_PAIRED <= 2
 0 <= aux_c_b5,r9 <= 536870910
 0 <= aux_c_b5,r8 <= 536870910
 0 <= aux_c_b5,r7 <= 536870910
 0 <= aux_c_b5,r6 <= 536870910
 0 <= aux_c_b5,r5 <= 536870910
 0 <= aux_c_b5,r4 <= 536870910
 0 <= aux_c_b5,r3 <= 536870910
 0 <= aux_cap_b9,r2,band_UNPAIRED <= 536870910
 0 <= aux_c_b5,r2 <= 536870910
 0 <= aux_c_b5,r1 <= 536870910
 0 <= aux_c_b5,r0 <= 536870910
 0 <= aux_cap_b6,r6,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_helper__b7,r12,band_HIGH_PAIRED__condX1 <= 3
 0 <= X__b8,r0,band_HIGH_PAIRED <= 3
 0 <= aux_cap_b0,r2,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b7,r12,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b7,r12,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b1,r2,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b1,r2,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b9,r7,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b1,r2,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b9,r7,band_LOW_PAIRED__condX2 <= 2
 0 <= sv_function_helpervar_b9,r6__condX2 <= 536870910
 0 <= aux_cap_b3,r7,band_HIGH_PAIRED <= 536870910
 0 <= sv_function_helpervar_b9,r6__condX3 <= 536870910
 0 <= sv_function_helpervar_b9,r6__condX1 <= 536870910
 0 <= aux_cap_helper__b5,r12,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b5,r12,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b6,r12,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b8,r7,band_LOW_PAIRED__Z2 <= 1
 0 <= X__b2,r13,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b6,r12,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b8,r7,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b8,r13,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_b0,r1,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b5,r13,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b8,r13,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b5,r13,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b5,r13,band_HIGH_PAIRED__condX1 <= 3
 0 <= sv_function_helpervar_b3,r12__condX3 <= 536870910
 0 <= sv_function_helpervar_b3,r12__condX2 <= 536870910
 0 <= sv_function_helpervar_b3,r12__condX1 <= 536870910
 0 <= aux_cap_helper__b2,r3,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b2,r3,band_UNPAIRED__Z2 <= 1
 0 <= X__b1,r7,band_HIGH_PAIRED <= 3
 0 <= X__b8,r9,band_HIGH_PAIRED <= 3
 0 <= sv_function_helpervar_b5,r12__Z2 <= 1
 0 <= sv_function_helpervar_b5,r12__Z1 <= 1
 0 <= aux_cap_b6,r0,band_UNPAIRED <= 536870910
 0 <= sv_function_helpervar_b5,r12__Z3 <= 1
 0 <= aux_cap_helper__b4,r7,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b4,r7,band_UNPAIRED__Z1 <= 1
 0 <= X__b3,r1,band_HIGH_PAIRED <= 3
 0 <= aux_cap_helper__b4,r0,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b4,r0,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b4,r0,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_b1,r7,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b5,r9,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_b9,r10,band_HIGH_PAIRED <= 536870910
 0 <= sv_function_helpervar_b0,r11__condX2 <= 536870910
 0 <= sv_function_helpervar_b0,r11__condX1 <= 536870910
 0 <= aux_cap_helper__b5,r9,band_LOW_PAIRED__condX1 <= 2
 0 <= sv_function_helpervar_b0,r11__condX3 <= 536870910
 0 <= aux_cap_b6,r7,band_UNPAIRED <= 536870910
 0 <= sv_function_helpervar_b3,r11__condX1 <= 536870910
 0 <= aux_cap_helper__b2,r12,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b6,r9,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b2,r12,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b6,r9,band_UNPAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b4,r6__condX3 <= 536870910
 0 <= sv_function_helpervar_b3,r11__condX2 <= 536870910
 0 <= sv_function_helpervar_b4,r6__condX1 <= 536870910
 0 <= sv_function_helpervar_b3,r11__condX3 <= 536870910
 0 <= sv_function_helpervar_b4,r6__condX2 <= 536870910
 0 <= aux_cap_helper__b7,r7,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b7,r7,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b5,r10,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b5,r10,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b5,r10,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b5,r7,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b6,r7,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b6,r7,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b7,r7,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b7,r7,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_b9,r10,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b5,r7,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b7,r7,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b0,r1,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_c_b3,r12 <= 536870910
 0 <= aux_cap_helper__b0,r1,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_c_b3,r11 <= 536870910
 0 <= sv_function_helpervar_b4,r12__Z1 <= 1
 0 <= aux_c_b3,r10 <= 536870910
 0 <= sv_function_helpervar_b4,r12__Z2 <= 1
 0 <= aux_cap_helper__b7,r9,band_UNPAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b4,r12__Z3 <= 1
 0 <= aux_cap_helper__b1,r5,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b1,r5,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b1,r5,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b7,r9,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_b8,r10,band_HIGH_PAIRED <= 536870910
 0 <= aux_c_b3,r13 <= 536870910
 0 <= sv_function_helpervar_b1,r0__Z2 <= 1
 0 <= sv_function_helpervar_b1,r0__Z3 <= 1
 0 <= sv_function_helpervar_b1,r0__Z1 <= 1
 0 <= aux_Omega_b4,r0 <= 536870910
 0 <= aux_cap_helper__b8,r9,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b8,r9,band_UNPAIRED__Z1 <= 1
 0 <= aux_sv_b2,r10 <= 536870910
 0 <= aux_sv_b2,r11 <= 536870910
 0 <= aux_cap_b0,r13,band_UNPAIRED <= 536870910
 0 <= aux_cap_b8,r5,band_HIGH_PAIRED <= 536870910
 0 <= aux_sv_b2,r12 <= 536870910
 0 <= aux_sv_b2,r13 <= 536870910
 0 <= aux_cap_helper__b2,r4,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b2,r4,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_b7,r10,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b7,r4,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b7,r4,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b7,r4,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b6,r5,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b6,r5,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b6,r11,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_b6,r1,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b6,r11,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b6,r11,band_HIGH_PAIRED__condX1 <= 3
 0 <= sv_function_helpervar_b6,r1__Z2 <= 1
 0 <= sv_function_helpervar_b6,r1__Z3 <= 1
 0 <= sv_function_helpervar_b6,r1__Z1 <= 1
 0 <= aux_cap_b3,r1,band_LOW_PAIRED <= 536870910
 0 <= aux_Omega_b4,r9 <= 536870910
 0 <= aux_Omega_b4,r8 <= 536870910
 0 <= aux_cap_b9,r4,band_LOW_PAIRED <= 536870910
 0 <= aux_Omega_b4,r7 <= 536870910
 0 <= aux_Omega_b4,r6 <= 536870910
 0 <= aux_Omega_b4,r5 <= 536870910
 0 <= aux_Omega_b4,r4 <= 536870910
 0 <= aux_Omega_b4,r3 <= 536870910
 0 <= X__b5,r0,band_LOW_PAIRED <= 2
 0 <= aux_Omega_b4,r2 <= 536870910
 0 <= aux_cap_b5,r1,band_UNPAIRED <= 536870910
 0 <= aux_Omega_b4,r1 <= 536870910
 0 <= aux_cap_helper__b4,r5,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b4,r5,band_UNPAIRED__Z2 <= 1
 0 <= sv_function_helpervar_b7,r4__condX1 <= 536870910
 0 <= sv_function_helpervar_b7,r4__condX2 <= 536870910
 0 <= X__b7,r5,band_HIGH_PAIRED <= 3
 0 <= aux_cap_b2,r13,band_UNPAIRED <= 536870910
 0 <= sv_function_helpervar_b7,r4__condX3 <= 536870910
 0 <= aux_cap_helper__b5,r3,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b5,r3,band_UNPAIRED__Z1 <= 1
 0 <= X__b6,r5,band_HIGH_PAIRED <= 3
 0 <= aux_cap_b5,r10,band_UNPAIRED <= 536870910
 0 <= X__b2,r4,band_LOW_PAIRED <= 2
 0 <= sv_function_helpervar_b9,r1__Z3 <= 1
 0 <= sv_function_helpervar_b9,r1__Z1 <= 1
 0 <= sv_function_helpervar_b9,r1__Z2 <= 1
 0 <= aux_c_b2,r9 <= 536870910
 0 <= aux_c_b2,r6 <= 536870910
 0 <= aux_cap_helper__b3,r1,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_c_b2,r5 <= 536870910
 0 <= aux_c_b2,r8 <= 536870910
 0 <= aux_c_b2,r7 <= 536870910
 0 <= aux_cap_helper__b3,r1,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_c_b2,r2 <= 536870910
 0 <= aux_c_b2,r1 <= 536870910
 0 <= aux_c_b2,r4 <= 536870910
 0 <= aux_c_b2,r3 <= 536870910
 0 <= aux_c_b2,r0 <= 536870910
 0 <= aux_cap_b4,r3,band_UNPAIRED <= 536870910
 0 <= sv_function_helpervar_b6,r8__condX3 <= 536870910
 0 <= sv_function_helpervar_b6,r8__condX1 <= 536870910
 0 <= sv_function_helpervar_b6,r8__condX2 <= 536870910
 0 <= X__b4,r3,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b7,r0,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b7,r0,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b5,r7,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b9,r7,band_UNPAIRED__condX2 <= 2
 0 <= X__b6,r8,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b9,r7,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b1,r3,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b1,r3,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_b0,r5,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_helper__b1,r3,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b3,r3,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b5,r7,band_UNPAIRED__Z2 <= 1
 0 <= X__b0,r13,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b3,r3,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b1,r0,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b1,r0,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b1,r11,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_b0,r9,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b1,r11,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b4,r9,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b4,r9,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b8,r8,band_HIGH_PAIRED__condX1 <= 3
 0 <= X__b6,r11,band_HIGH_PAIRED <= 3
 0 <= aux_cap_helper__b7,r8,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b8,r8,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b7,r8,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b8,r8,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b3,r6,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b0,r13,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b3,r6,band_UNPAIRED__condX2 <= 2
 0 <= sv_function_helpervar_b4,r0__Z3 <= 1
 0 <= aux_cap_helper__b5,r6,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b5,r6,band_LOW_PAIRED__Z1 <= 1
 0 <= X__b3,r5,band_UNPAIRED <= 2
 0 <= aux_cap_b2,r1,band_LOW_PAIRED <= 536870910
 0 <= GlobalBidder_W_ir__b7,r3 <= 1
 0 <= GlobalBidder_W_ir__b7,r2 <= 1
 0 <= aux_cap_helper__b0,r13,band_HIGH_PAIRED__condX2 <= 3
 0 <= GlobalBidder_W_ir__b7,r1 <= 1
 0 <= aux_cap_helper__b0,r13,band_HIGH_PAIRED__condX3 <= 3
 0 <= GlobalBidder_W_ir__b7,r0 <= 1
 0 <= X__b1,r8,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b2,r1,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b2,r1,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_b1,r12,band_LOW_PAIRED <= 536870910
 0 <= sv_function_helpervar_b9,r10__Z3 <= 1
 0 <= sv_function_helpervar_b9,r10__Z1 <= 1
 0 <= sv_function_helpervar_b9,r10__Z2 <= 1
 0 <= X__b7,r11,band_HIGH_PAIRED <= 3
 0 <= aux_cap_b1,r3,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_helper__b6,r2,band_LOW_PAIRED__condX2 <= 2
 0 <= X__b8,r7,band_LOW_PAIRED <= 2
 0 <= aux_cap_helper__b0,r6,band_UNPAIRED__condX2 <= 2
 0 <= aux_Omega_b6,r10 <= 536870910
 0 <= aux_Omega_b6,r11 <= 536870910
 0 <= aux_cap_helper__b3,r1,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b3,r1,band_UNPAIRED__Z1 <= 1
 0 <= aux_Omega_b6,r12 <= 536870910
 0 <= aux_Omega_b6,r13 <= 536870910
 0 <= sv_function_helpervar_b2,r4__condX3 <= 536870910
 0 <= aux_cap_b3,r11,band_HIGH_PAIRED <= 536870910
 0 <= sv_function_helpervar_b2,r4__condX1 <= 536870910
 0 <= sv_function_helpervar_b2,r4__condX2 <= 536870910
 0 <= aux_cap_helper__b0,r2,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b0,r2,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b0,r6,band_UNPAIRED__condX1 <= 2
 0 <= sv_function_helpervar_b3,r5__Z3 <= 1
 0 <= sv_function_helpervar_b3,r5__Z2 <= 1
 0 <= sv_function_helpervar_b3,r5__Z1 <= 1
 0 <= sv_function_helpervar_b2,r11__Z3 <= 1
 0 <= aux_cap_helper__b5,r10,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b5,r10,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_b5,r9,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b6,r2,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b6,r1,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b6,r1,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_b0,r10,band_LOW_PAIRED <= 536870910
 0 <= sv_function_helpervar_b5,r2__condX2 <= 536870910
 0 <= sv_function_helpervar_b5,r2__condX3 <= 536870910
 0 <= sv_function_helpervar_b5,r2__condX1 <= 536870910
 0 <= aux_cap_b2,r11,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b7,r12,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b6,r11,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b6,r11,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b6,r12,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b6,r12,band_LOW_PAIRED__Z1 <= 1
 0 <= GlobalBidder_W_ir__b7,r7 <= 1
 0 <= GlobalBidder_W_ir__b7,r6 <= 1
 0 <= sv_function_helpervar_b2,r11__Z2 <= 1
 0 <= GlobalBidder_W_ir__b7,r5 <= 1
 0 <= sv_function_helpervar_b2,r11__Z1 <= 1
 0 <= GlobalBidder_W_ir__b7,r4 <= 1
 0 <= aux_cap_b8,r4,band_LOW_PAIRED <= 536870910
 0 <= GlobalBidder_W_ir__b7,r9 <= 1
 0 <= GlobalBidder_W_ir__b7,r8 <= 1
 0 <= aux_cap_helper__b2,r7,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b4,r3,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b2,r7,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b4,r3,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b7,r12,band_LOW_PAIRED__condX2 <= 2
 0 <= sv_function_helpervar_b4,r0__Z1 <= 1
 0 <= sv_function_helpervar_b4,r0__Z2 <= 1
 0 <= X__b4,r0,band_LOW_PAIRED <= 2
 0 <= aux_cap_helper__b7,r1,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b7,r1,band_HIGH_PAIRED__condX3 <= 3
 0 <= X__b9,r9,band_HIGH_PAIRED <= 3
 0 <= aux_cap_helper__b7,r1,band_HIGH_PAIRED__condX1 <= 3
 0 <= X__b4,r1,band_HIGH_PAIRED <= 3
 0 <= aux_cap_helper__b8,r6,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b8,r6,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b2,r3,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b2,r3,band_UNPAIRED__condX1 <= 2
 0 <= X__b9,r3,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b5,r0,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b5,r0,band_HIGH_PAIRED__Z2 <= 1
 0 <= X__b5,r10,band_UNPAIRED <= 2
 0 <= aux_cap_b7,r6,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_helper__b0,r3,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_b5,r6,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b0,r3,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b4,r3,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b4,r3,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b2,r13,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b2,r13,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b5,r5,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b5,r5,band_UNPAIRED__Z2 <= 1
 0 <= X__b7,r0,band_HIGH_PAIRED <= 3
 0 <= aux_cap_helper__b2,r13,band_HIGH_PAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b1,r8__condX2 <= 536870910
 0 <= sv_function_helpervar_b1,r8__condX3 <= 536870910
 0 <= aux_cap_helper__b5,r0,band_HIGH_PAIRED__Z3 <= 1
 0 <= sv_function_helpervar_b1,r8__condX1 <= 536870910
 0 <= aux_cap_helper__b1,r4,band_HIGH_PAIRED__Z1 <= 1
 0 <= X__b0,r7,band_HIGH_PAIRED <= 3
 0 <= aux_cap_helper__b1,r4,band_HIGH_PAIRED__Z2 <= 1
 0 <= sv_function_helpervar_b7,r13__Z1 <= 1
 0 <= aux_cap_helper__b1,r13,band_UNPAIRED__Z2 <= 1
 0 <= sv_function_helpervar_b7,r13__Z2 <= 1
 0 <= aux_cap_helper__b1,r13,band_UNPAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b7,r13__Z3 <= 1
 0 <= aux_cap_helper__b4,r6,band_HIGH_PAIRED__condX1 <= 3
 0 <= X__b3,r2,band_LOW_PAIRED <= 2
 0 <= aux_cap_helper__b1,r4,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b4,r6,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b4,r6,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b3,r2,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b3,r2,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b4,r13,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b3,r2,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b4,r13,band_LOW_PAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b8,r6__Z3 <= 1
 0 <= X__b9,r5,band_LOW_PAIRED <= 2
 0 <= sv_function_helpervar_b8,r6__Z2 <= 1
 0 <= sv_function_helpervar_b8,r6__Z1 <= 1
 0 <= X__b0,r2,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b5,r12,band_UNPAIRED__condX2 <= 2
 0 <= sv_function_helpervar_b9,r12__condX2 <= 536870910
 0 <= sv_function_helpervar_b9,r12__condX1 <= 536870910
 0 <= aux_cap_b2,r7,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b5,r12,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b9,r1,band_LOW_PAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b9,r12__condX3 <= 536870910
 0 <= X__b6,r0,band_UNPAIRED <= 2
 0 <= aux_cap_b8,r4,band_UNPAIRED <= 536870910
 0 <= aux_cap_b9,r0,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b9,r1,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b1,r9,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b4,r0,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b4,r0,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b1,r9,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_b7,r10,band_UNPAIRED <= 536870910
 0 <= aux_cap_b2,r6,band_UNPAIRED <= 536870910
 0 <= X__b1,r12,band_HIGH_PAIRED <= 3
 0 <= sv_function_helpervar_b3,r0__Z3 <= 1
 0 <= aux_cap_helper__b6,r9,band_UNPAIRED__condX1 <= 2
 0 <= sv_function_helpervar_b3,r0__Z2 <= 1
 0 <= aux_cap_helper__b6,r9,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b9,r10,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b9,r10,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b9,r1,band_HIGH_PAIRED__Z1 <= 1
 0 <= X__b2,r8,band_LOW_PAIRED <= 2
 0 <= aux_cap_helper__b9,r1,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b9,r1,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b7,r13,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b7,r13,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_b1,r8,band_HIGH_PAIRED <= 536870910
 0 <= X__b5,r5,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b0,r3,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b0,r3,band_LOW_PAIRED__Z2 <= 1
 0 <= X__b6,r4,band_HIGH_PAIRED <= 3
 0 <= aux_Omega_b9,r1 <= 536870910
 0 <= aux_Omega_b9,r0 <= 536870910
 0 <= sv_function_helpervar_b0,r2__Z2 <= 1
 0 <= aux_Omega_b9,r3 <= 536870910
 0 <= sv_function_helpervar_b0,r2__Z1 <= 1
 0 <= aux_Omega_b9,r2 <= 536870910
 0 <= aux_Omega_b9,r5 <= 536870910
 0 <= sv_function_helpervar_b0,r2__Z3 <= 1
 0 <= aux_Omega_b9,r4 <= 536870910
 0 <= aux_cap_b2,r8,band_UNPAIRED <= 536870910
 0 <= X__b4,r1,band_LOW_PAIRED <= 2
 0 <= aux_cap_helper__b5,r1,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b5,r1,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b5,r1,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_b6,r0,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b2,r2,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b7,r10,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b2,r2,band_HIGH_PAIRED__condX3 <= 3
 0 <= sv_function_helpervar_b3,r0__condX1 <= 536870910
 0 <= aux_cap_helper__b7,r11,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b2,r2,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b7,r11,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b2,r1,band_LOW_PAIRED__condX2 <= 2
 0 <= GlobalBidder_W_ir__b9,r6 <= 1
 0 <= aux_cap_helper__b2,r1,band_LOW_PAIRED__condX1 <= 2
 0 <= GlobalBidder_W_ir__b9,r7 <= 1
 0 <= sv_function_helpervar_b3,r0__condX2 <= 536870910
 0 <= aux_cap_helper__b7,r10,band_HIGH_PAIRED__condX3 <= 3
 0 <= GlobalBidder_W_ir__b9,r8 <= 1
 0 <= sv_function_helpervar_b3,r0__condX3 <= 536870910
 0 <= aux_c_b6,r9 <= 536870910
 0 <= aux_cap_helper__b7,r10,band_HIGH_PAIRED__condX2 <= 3
 0 <= GlobalBidder_W_ir__b9,r9 <= 1
 0 <= aux_c_b6,r8 <= 536870910
 0 <= aux_c_b6,r7 <= 536870910
 0 <= aux_c_b6,r6 <= 536870910
 0 <= aux_c_b6,r5 <= 536870910
 0 <= aux_c_b6,r4 <= 536870910
 0 <= aux_c_b6,r3 <= 536870910
 0 <= sv_function_helpervar_b3,r0__Z1 <= 1
 0 <= aux_c_b6,r2 <= 536870910
 0 <= aux_c_b6,r1 <= 536870910
 0 <= aux_c_b6,r0 <= 536870910
 0 <= aux_cap_helper__b8,r4,band_LOW_PAIRED__Z2 <= 1
 0 <= sv_function_helpervar_b3,r6__condX1 <= 536870910
 0 <= sv_function_helpervar_b3,r6__condX2 <= 536870910
 0 <= sv_function_helpervar_b3,r6__condX3 <= 536870910
 0 <= aux_cap_helper__b8,r4,band_LOW_PAIRED__Z1 <= 1
 0 <= X__b3,r10,band_LOW_PAIRED <= 2
 0 <= aux_cap_b4,r6,band_HIGH_PAIRED <= 536870910
 0 <= sv_function_helpervar_b1,r7__condX2 <= 536870910
 0 <= aux_cap_helper__b2,r6,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_Omega_b9,r7 <= 536870910
 0 <= sv_function_helpervar_b1,r7__condX1 <= 536870910
 0 <= aux_Omega_b9,r6 <= 536870910
 0 <= aux_cap_helper__b0,r1,band_UNPAIRED__condX1 <= 2
 0 <= aux_Omega_b9,r9 <= 536870910
 0 <= aux_cap_helper__b0,r1,band_UNPAIRED__condX2 <= 2
 0 <= sv_function_helpervar_b1,r7__condX3 <= 536870910
 0 <= aux_Omega_b9,r8 <= 536870910
 0 <= aux_cap_helper__b6,r5,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b2,r6,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b6,r5,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b0,r7,band_HIGH_PAIRED__condX3 <= 3
 0 <= X__b3,r6,band_HIGH_PAIRED <= 3
 0 <= aux_cap_helper__b6,r5,band_HIGH_PAIRED__condX3 <= 3
 0 <= X__b6,r6,band_LOW_PAIRED <= 2
 0 <= aux_cap_helper__b0,r7,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b0,r7,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b5,r8,band_UNPAIRED__condX2 <= 2
 0 <= X__b0,r3,band_LOW_PAIRED <= 2
 0 <= aux_cap_helper__b5,r8,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_b5,r8,band_LOW_PAIRED <= 536870910
 0 <= aux_sv_b6,r12 <= 536870910
 0 <= aux_sv_b6,r13 <= 536870910
 0 <= aux_cap_helper__b6,r11,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b6,r11,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b3,r13,band_UNPAIRED__condX2 <= 2
 0 <= aux_sv_b6,r10 <= 536870910
 0 <= aux_cap_helper__b3,r13,band_UNPAIRED__condX1 <= 2
 0 <= aux_sv_b6,r11 <= 536870910
 0 <= aux_cap_b1,r0,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b8,r3,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_b9,r9,band_UNPAIRED <= 536870910
 0 <= sv_function_helpervar_b6,r12__Z3 <= 1
 0 <= sv_function_helpervar_b6,r12__Z2 <= 1
 0 <= aux_cap_helper__b6,r11,band_HIGH_PAIRED__Z1 <= 1
 0 <= X__b9,r2,band_HIGH_PAIRED <= 3
 0 <= sv_function_helpervar_b6,r12__Z1 <= 1
 0 <= aux_cap_b3,r2,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b8,r0,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b8,r0,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b8,r0,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b8,r3,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b8,r3,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_b6,r2,band_LOW_PAIRED <= 536870910
 0 <= X__b6,r9,band_UNPAIRED <= 2
 0 <= sv_function_helpervar_b8,r3__Z1 <= 1
 0 <= sv_function_helpervar_b8,r3__Z3 <= 1
 0 <= sv_function_helpervar_b8,r3__Z2 <= 1
 0 <= X__b3,r11,band_UNPAIRED <= 2
 0 <= sv_function_helpervar_b1,r1__condX2 <= 536870910
 0 <= aux_cap_helper__b8,r10,band_LOW_PAIRED__condX1 <= 2
 0 <= sv_function_helpervar_b1,r1__condX1 <= 536870910
 0 <= aux_cap_helper__b8,r10,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_b1,r2,band_UNPAIRED <= 536870910
 0 <= sv_function_helpervar_b7,r6__Z2 <= 1
 0 <= sv_function_helpervar_b7,r6__Z3 <= 1
 0 <= aux_cap_helper__b3,r2,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b9,r9,band_LOW_PAIRED__Z2 <= 1
 0 <= sv_function_helpervar_b1,r1__condX3 <= 536870910
 0 <= aux_cap_helper__b3,r2,band_LOW_PAIRED__condX1 <= 2
 0 <= sv_function_helpervar_b7,r6__Z1 <= 1
 0 <= aux_cap_helper__b9,r10,band_HIGH_PAIRED__Z2 <= 1
 0 <= sv_function_helpervar_b2,r7__Z1 <= 1
 0 <= aux_cap_helper__b9,r10,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b9,r10,band_HIGH_PAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b2,r7__Z2 <= 1
 0 <= sv_function_helpervar_b2,r7__Z3 <= 1
 0 <= aux_cap_helper__b4,r0,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b4,r0,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b9,r4,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b4,r0,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b9,r4,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b9,r4,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b1,r1,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b1,r10,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b4,r10,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b1,r10,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b1,r1,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_c_b5,r10 <= 536870910
 0 <= aux_cap_helper__b4,r10,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b0,r13,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b4,r10,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b0,r13,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b0,r13,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_b1,r4,band_UNPAIRED <= 536870910
 0 <= aux_c_b5,r13 <= 536870910
 0 <= aux_c_b5,r12 <= 536870910
 0 <= aux_c_b5,r11 <= 536870910
 0 <= sv_function_helpervar_b5,r5__condX3 <= 536870910
 0 <= X__b2,r3,band_HIGH_PAIRED <= 3
 0 <= aux_cap_b4,r9,band_LOW_PAIRED <= 536870910
 0 <= X__b5,r7,band_LOW_PAIRED <= 2
 0 <= sv_function_helpervar_b5,r5__condX2 <= 536870910
 0 <= sv_function_helpervar_b5,r5__condX1 <= 536870910
 0 <= X__b4,r1,band_UNPAIRED <= 2
 0 <= X__b7,r8,band_HIGH_PAIRED <= 3
 0 <= aux_cap_helper__b5,r1,band_UNPAIRED__Z2 <= 1
 0 <= X__b1,r10,band_HIGH_PAIRED <= 3
 0 <= aux_cap_helper__b5,r1,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_b7,r4,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b2,r4,band_UNPAIRED__Z2 <= 1
 0 <= sv_function_helpervar_b2,r10__condX3 <= 536870910
 0 <= aux_cap_b5,r12,band_HIGH_PAIRED <= 536870910
 0 <= sv_function_helpervar_b2,r10__condX2 <= 536870910
 0 <= sv_function_helpervar_b2,r10__condX1 <= 536870910
 0 <= X__b0,r8,band_HIGH_PAIRED <= 3
 0 <= sv_function_helpervar_b6,r10__condX3 <= 536870910
 0 <= sv_function_helpervar_b6,r10__condX2 <= 536870910
 0 <= sv_function_helpervar_b6,r10__condX1 <= 536870910
 0 <= X__b2,r2,band_HIGH_PAIRED <= 3
 0 <= aux_cap_helper__b2,r4,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b5,r8,band_LOW_PAIRED__condX1 <= 2
 0 <= X__b5,r3,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b5,r8,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_b0,r10,band_HIGH_PAIRED <= 536870910
 0 <= aux_Omega_b2,r10 <= 536870910
 0 <= aux_Omega_b2,r11 <= 536870910
 0 <= X__b5,r11,band_LOW_PAIRED <= 2
 0 <= sv_function_helpervar_b0,r12__Z2 <= 1
 0 <= sv_function_helpervar_b0,r12__Z3 <= 1
 0 <= X__b9,r11,band_HIGH_PAIRED <= 3
 0 <= sv_function_helpervar_b0,r12__Z1 <= 1
 0 <= aux_cap_helper__b4,r9,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b4,r9,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_b7,r1,band_LOW_PAIRED <= 536870910
 0 <= X__b7,r10,band_UNPAIRED <= 2
 0 <= aux_Omega_b2,r12 <= 536870910
 0 <= aux_Omega_b2,r13 <= 536870910
 0 <= X__b7,r0,band_LOW_PAIRED <= 2
 0 <= X__b1,r9,band_LOW_PAIRED <= 2
 0 <= aux_cap_b8,r11,band_HIGH_PAIRED <= 536870910
 0 <= sv_function_helpervar_b0,r13__condX1 <= 536870910
 0 <= sv_function_helpervar_b0,r13__condX2 <= 536870910
 0 <= aux_cap_helper__b8,r4,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b5,r6,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b5,r6,band_UNPAIRED__condX2 <= 2
 0 <= sv_function_helpervar_b0,r13__condX3 <= 536870910
 0 <= sv_function_helpervar_b4,r13__condX3 <= 536870910
 0 <= sv_function_helpervar_b4,r13__condX1 <= 536870910
 0 <= sv_function_helpervar_b4,r13__condX2 <= 536870910
 0 <= aux_cap_helper__b6,r11,band_UNPAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b8,r13__condX1 <= 536870910
 0 <= sv_function_helpervar_b8,r13__condX2 <= 536870910
 0 <= sv_function_helpervar_b8,r13__condX3 <= 536870910
 0 <= X__b0,r12,band_LOW_PAIRED <= 2
 0 <= X__b6,r12,band_HIGH_PAIRED <= 3
 0 <= aux_cap_helper__b6,r11,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_b2,r9,band_LOW_PAIRED <= 536870910
 0 <= sv_function_helpervar_b4,r13__Z1 <= 1
 0 <= sv_function_helpervar_b4,r13__Z2 <= 1
 0 <= X__b5,r9,band_UNPAIRED <= 2
 0 <= sv_function_helpervar_b4,r13__Z3 <= 1
 0 <= X__b5,r1,band_HIGH_PAIRED <= 3
 0 <= sv_function_helpervar_b5,r5__Z1 <= 1
 0 <= sv_function_helpervar_b5,r5__Z2 <= 1
 0 <= aux_cap_helper__b4,r12,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b4,r12,band_LOW_PAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b5,r5__Z3 <= 1
 0 <= aux_cap_helper__b9,r2,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b9,r2,band_LOW_PAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b8,r4__condX3 <= 536870910
 0 <= sv_function_helpervar_b8,r4__condX1 <= 536870910
 0 <= sv_function_helpervar_b8,r4__condX2 <= 536870910
 0 <= X__b5,r7,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b6,r7,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b6,r7,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_b4,r13,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b1,r10,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b1,r10,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_b0,r0,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b1,r10,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b0,r6,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b0,r6,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b3,r6,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b6,r4,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b3,r6,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b6,r4,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b6,r4,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b3,r6,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b9,r5,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b9,r5,band_UNPAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b6,r9__Z3 <= 1
 0 <= sv_function_helpervar_b6,r9__Z2 <= 1
 0 <= sv_function_helpervar_b6,r9__Z1 <= 1
 0 <= aux_cap_b2,r13,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b1,r6,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b1,r6,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b1,r6,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b5,r9,band_UNPAIRED__Z2 <= 1
 0 <= sv_function_helpervar_b1,r11__condX1 <= 536870910
 0 <= sv_function_helpervar_b1,r11__condX2 <= 536870910
 0 <= sv_function_helpervar_b1,r11__condX3 <= 536870910
 0 <= aux_cap_helper__b5,r9,band_UNPAIRED__Z1 <= 1
 0 <= X__b9,r1,band_HIGH_PAIRED <= 3
 0 <= X__b4,r9,band_HIGH_PAIRED <= 3
 0 <= aux_cap_b7,r3,band_UNPAIRED <= 536870910
 0 <= aux_cap_b7,r8,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_helper__b8,r6,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b8,r6,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b1,r7,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_b0,r12,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b1,r7,band_LOW_PAIRED__condX2 <= 2
 0 <= sv_function_helpervar_b2,r0__condX1 <= 536870910
 0 <= sv_function_helpervar_b2,r0__condX2 <= 536870910
 0 <= sv_function_helpervar_b2,r0__condX3 <= 536870910
 0 <= sv_function_helpervar_b9,r7__Z3 <= 1
 0 <= sv_function_helpervar_b9,r7__Z1 <= 1
 0 <= sv_function_helpervar_b9,r7__Z2 <= 1
 0 <= sv_function_helpervar_b9,r11__condX1 <= 536870910
 0 <= aux_cap_b4,r11,band_UNPAIRED <= 536870910
 0 <= sv_function_helpervar_b9,r11__condX3 <= 536870910
 0 <= aux_cap_b6,r11,band_LOW_PAIRED <= 536870910
 0 <= sv_function_helpervar_b9,r11__condX2 <= 536870910
 0 <= aux_cap_b8,r0,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_b9,r5,band_UNPAIRED <= 536870910
 0 <= aux_cap_b3,r13,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_helper__b4,r7,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b4,r7,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b4,r11,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b4,r11,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b3,r0,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b3,r0,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b7,r8,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b7,r8,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b8,r0,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b8,r0,band_LOW_PAIRED__condX1 <= 2
 0 <= X__b6,r5,band_UNPAIRED <= 2
 0 <= X__b6,r13,band_HIGH_PAIRED <= 3
 0 <= aux_cap_helper__b6,r3,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_b8,r8,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_b9,r1,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_helper__b6,r3,band_HIGH_PAIRED__condX2 <= 3
 0 <= X__b6,r13,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b6,r3,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_b8,r9,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b8,r4,band_LOW_PAIRED__condX2 <= 2
 0 <= sv_function_helpervar_b4,r0__condX2 <= 536870910
 0 <= sv_function_helpervar_b4,r0__condX1 <= 536870910
 0 <= X__b5,r1,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b3,r6,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b3,r6,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b5,r9,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b5,r9,band_HIGH_PAIRED__Z3 <= 1
 0 <= sv_function_helpervar_b4,r0__condX3 <= 536870910
 0 <= aux_cap_helper__b5,r9,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b7,r2,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b6,r3,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b6,r3,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b3,r4,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b7,r2,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b8,r9,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b8,r13,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b3,r4,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b7,r2,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b8,r9,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b7,r10,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_b1,r9,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_b1,r5,band_LOW_PAIRED <= 536870910
 0 <= X__b2,r1,band_LOW_PAIRED <= 2
 0 <= X__b3,r13,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b7,r10,band_HIGH_PAIRED__Z2 <= 1
 0 <= sv_function_helpervar_b2,r3__Z1 <= 1
 0 <= aux_cap_helper__b7,r10,band_HIGH_PAIRED__Z3 <= 1
 0 <= X__b6,r1,band_LOW_PAIRED <= 2
 0 <= aux_cap_b0,r11,band_HIGH_PAIRED <= 536870910
 0 <= sv_function_helpervar_b2,r3__Z2 <= 1
 0 <= sv_function_helpervar_b2,r3__Z3 <= 1
 0 <= aux_cap_helper__b7,r6,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b7,r6,band_LOW_PAIRED__Z2 <= 1
 0 <= sv_function_helpervar_b1,r11__Z1 <= 1
 0 <= sv_function_helpervar_b1,r11__Z3 <= 1
 0 <= sv_function_helpervar_b1,r11__Z2 <= 1
 0 <= aux_cap_helper__b1,r9,band_HIGH_PAIRED__condX3 <= 3
 0 <= X__b0,r9,band_HIGH_PAIRED <= 3
 0 <= aux_cap_helper__b1,r9,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b4,r12,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b1,r9,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b4,r12,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b4,r12,band_HIGH_PAIRED__condX1 <= 3
 0 <= X__b6,r7,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b6,r5,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b6,r5,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_b9,r11,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b9,r8,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b9,r8,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_b7,r13,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_helper__b5,r11,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b5,r11,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_b4,r2,band_LOW_PAIRED <= 536870910
 0 <= sv_function_helpervar_b7,r2__Z2 <= 1
 0 <= sv_function_helpervar_b7,r2__Z3 <= 1
 0 <= X__b1,r0,band_HIGH_PAIRED <= 3
 0 <= sv_function_helpervar_b7,r2__Z1 <= 1
 0 <= aux_cap_b9,r7,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b8,r13,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b8,r13,band_HIGH_PAIRED__condX2 <= 3
 0 <= X__b6,r13,band_LOW_PAIRED <= 2
 0 <= aux_cap_b2,r4,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_helper__b8,r2,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b8,r2,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b8,r2,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b9,r5,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_b7,r1,band_UNPAIRED <= 536870910
 0 <= aux_cap_b9,r13,band_UNPAIRED <= 536870910
 0 <= X__b9,r13,band_LOW_PAIRED <= 2
 0 <= X__b6,r3,band_UNPAIRED <= 2
 0 <= aux_cap_b8,r7,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b8,r0,band_HIGH_PAIRED__Z3 <= 1
 0 <= sv_function_helpervar_b9,r4__condX1 <= 536870910
 0 <= aux_cap_helper__b8,r0,band_HIGH_PAIRED__Z2 <= 1
 0 <= X__b8,r4,band_LOW_PAIRED <= 2
 0 <= X__b6,r11,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b8,r0,band_HIGH_PAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b9,r4__condX3 <= 536870910
 0 <= aux_cap_b9,r6,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_helper__b9,r5,band_LOW_PAIRED__condX1 <= 2
 0 <= sv_function_helpervar_b9,r4__condX2 <= 536870910
 0 <= aux_cap_helper__b8,r11,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b8,r11,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_b8,r7,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b8,r11,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b0,r5,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b6,r0,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b0,r5,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b0,r5,band_HIGH_PAIRED__condX2 <= 3
 0 <= X__b7,r9,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b1,r5,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b3,r12,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b0,r6,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b3,r12,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_b4,r7,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b0,r6,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b6,r0,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b1,r5,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b3,r12,band_HIGH_PAIRED__Z1 <= 1
 0 <= X__b2,r13,band_LOW_PAIRED <= 2
 0 <= aux_cap_helper__b7,r6,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b7,r6,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b0,r8,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_b0,r13,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_helper__b0,r8,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_b8,r1,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b0,r8,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_b8,r7,band_LOW_PAIRED <= 536870910
 0 <= X__b5,r0,band_HIGH_PAIRED <= 3
 0 <= sv_function_helpervar_b9,r9__condX1 <= 536870910
 0 <= sv_function_helpervar_b9,r9__condX2 <= 536870910
 0 <= sv_function_helpervar_b3,r10__Z3 <= 1
 0 <= sv_function_helpervar_b0,r1__condX1 <= 536870910
 0 <= sv_function_helpervar_b0,r1__condX3 <= 536870910
 0 <= aux_cap_helper__b7,r2,band_UNPAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b0,r1__condX2 <= 536870910
 0 <= aux_cap_helper__b7,r2,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b2,r7,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b2,r7,band_UNPAIRED__Z2 <= 1
 0 <= sv_function_helpervar_b9,r9__condX3 <= 536870910
 0 <= sv_function_helpervar_b5,r1__Z3 <= 1
 0 <= aux_cap_helper__b1,r0,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b1,r0,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b3,r6,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b3,r6,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_b3,r3,band_LOW_PAIRED <= 536870910
 0 <= sv_function_helpervar_b8,r11__Z1 <= 1
 0 <= sv_function_helpervar_b8,r11__Z2 <= 1
 0 <= sv_function_helpervar_b5,r1__Z2 <= 1
 0 <= sv_function_helpervar_b8,r11__Z3 <= 1
 0 <= sv_function_helpervar_b5,r1__Z1 <= 1
 0 <= aux_sv_b7,r5 <= 536870910
 0 <= aux_sv_b7,r4 <= 536870910
 0 <= aux_sv_b7,r7 <= 536870910
 0 <= aux_sv_b7,r6 <= 536870910
 0 <= aux_sv_b7,r9 <= 536870910
 0 <= aux_cap_b8,r3,band_UNPAIRED <= 536870910
 0 <= aux_sv_b7,r8 <= 536870910
 0 <= X__b9,r10,band_HIGH_PAIRED <= 3
 0 <= aux_cap_helper__b4,r2,band_LOW_PAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b3,r10__Z1 <= 1
 0 <= aux_cap_helper__b4,r2,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b6,r10,band_LOW_PAIRED__Z2 <= 1
 0 <= sv_function_helpervar_b3,r10__Z2 <= 1
 0 <= aux_cap_helper__b6,r10,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_sv_b7,r1 <= 536870910
 0 <= aux_sv_b7,r0 <= 536870910
 0 <= aux_sv_b7,r3 <= 536870910
 0 <= aux_sv_b7,r2 <= 536870910
 0 <= X__b2,r1,band_HIGH_PAIRED <= 3
 0 <= aux_cap_b8,r5,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b5,r12,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b5,r12,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_b5,r9,band_HIGH_PAIRED <= 536870910
 0 <= X__b1,r2,band_LOW_PAIRED <= 2
 0 <= X__b7,r5,band_LOW_PAIRED <= 2
 0 <= X__b5,r2,band_LOW_PAIRED <= 2
 0 <= sv_function_helpervar_b5,r12__condX3 <= 536870910
 0 <= sv_function_helpervar_b5,r12__condX2 <= 536870910
 0 <= sv_function_helpervar_b5,r12__condX1 <= 536870910
 0 <= aux_cap_helper__b8,r8,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b8,r8,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_c_b9,r13 <= 536870910
 0 <= aux_c_b9,r12 <= 536870910
 0 <= aux_cap_helper__b9,r12,band_UNPAIRED__Z2 <= 1
 0 <= aux_c_b9,r11 <= 536870910
 0 <= aux_cap_helper__b3,r8,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_c_b9,r10 <= 536870910
 0 <= aux_cap_helper__b9,r12,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b3,r8,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b3,r8,band_HIGH_PAIRED__condX2 <= 3
 0 <= X__b7,r9,band_HIGH_PAIRED <= 3
 0 <= aux_cap_b0,r10,band_UNPAIRED <= 536870910
 0 <= sv_function_helpervar_b4,r8__Z2 <= 1
 0 <= sv_function_helpervar_b4,r8__Z1 <= 1
 0 <= aux_cap_b0,r12,band_HIGH_PAIRED <= 536870910
 0 <= sv_function_helpervar_b4,r8__Z3 <= 1
 0 <= sv_function_helpervar_b6,r4__condX1 <= 536870910
 0 <= sv_function_helpervar_b6,r4__condX2 <= 536870910
 0 <= sv_function_helpervar_b6,r4__condX3 <= 536870910
 0 <= X__b5,r2,band_HIGH_PAIRED <= 3
 0 <= aux_cap_b0,r6,band_HIGH_PAIRED <= 536870910
 0 <= X__b6,r1,band_UNPAIRED <= 2
 0 <= aux_cap_b8,r9,band_HIGH_PAIRED <= 536870910
 0 <= v__9 <= 536870910
 0 <= v__8 <= 536870910
 0 <= v__7 <= 536870910
 0 <= v__6 <= 536870910
 0 <= X__b7,r6,band_HIGH_PAIRED <= 3
 0 <= aux_cap_helper__b0,r11,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b0,r11,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b9,r10,band_LOW_PAIRED__condX1 <= 2
 0 <= v__5 <= 536870910
 0 <= v__4 <= 536870910
 0 <= v__3 <= 536870910
 0 <= v__2 <= 536870910
 0 <= v__1 <= 536870910
 0 <= v__0 <= 536870910
 0 <= X__b3,r3,band_UNPAIRED <= 2
 0 <= GlobalBidder_PSI__i9,k2 <= 536870910
 0 <= X__b0,r4,band_UNPAIRED <= 2
 0 <= GlobalBidder_PSI__i9,k1 <= 536870910
 0 <= GlobalBidder_PSI__i9,k0 <= 536870910
 0 <= aux_cap_b0,r9,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_b9,r1,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b9,r10,band_LOW_PAIRED__condX2 <= 2
 0 <= X__b2,r11,band_LOW_PAIRED <= 2
 0 <= aux_cap_helper__b9,r4,band_HIGH_PAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b7,r5__condX3 <= 536870910
 0 <= aux_cap_helper__b9,r4,band_HIGH_PAIRED__Z2 <= 1
 0 <= sv_function_helpervar_b7,r5__condX2 <= 536870910
 0 <= sv_function_helpervar_b7,r5__condX1 <= 536870910
 0 <= aux_cap_helper__b9,r4,band_HIGH_PAIRED__Z3 <= 1
 0 <= sv_function_helpervar_b6,r6__Z2 <= 1
 0 <= aux_cap_helper__b3,r7,band_LOW_PAIRED__Z2 <= 1
 0 <= sv_function_helpervar_b6,r6__Z1 <= 1
 0 <= aux_cap_helper__b3,r7,band_LOW_PAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b6,r6__Z3 <= 1
 0 <= X__b4,r9,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b2,r11,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b2,r11,band_UNPAIRED__Z2 <= 1
 0 <= GlobalBidder_W_i_9 <= 14
 0 <= GlobalBidder_W_i_8 <= 14
 0 <= sv_function_helpervar_b5,r13__Z3 <= 1
 0 <= GlobalBidder_W_i_7 <= 14
 0 <= aux_cap_helper__b9,r6,band_LOW_PAIRED__Z2 <= 1
 0 <= sv_function_helpervar_b5,r13__Z2 <= 1
 0 <= aux_cap_helper__b9,r6,band_LOW_PAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b5,r13__Z1 <= 1
 0 <= aux_cap_b4,r4,band_LOW_PAIRED <= 536870910
 0 <= X__b3,r9,band_LOW_PAIRED <= 2
 0 <= aux_sv_b1,r4 <= 536870910
 0 <= aux_sv_b1,r5 <= 536870910
 0 <= X__b9,r0,band_LOW_PAIRED <= 2
 0 <= aux_sv_b1,r2 <= 536870910
 0 <= aux_sv_b1,r3 <= 536870910
 0 <= aux_sv_b1,r0 <= 536870910
 0 <= aux_sv_b1,r1 <= 536870910
 0 <= sv_function_helpervar_b9,r4__Z1 <= 1
 0 <= sv_function_helpervar_b9,r4__Z2 <= 1
 0 <= sv_function_helpervar_b9,r4__Z3 <= 1
 0 <= aux_cap_b9,r8,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_helper__b7,r12,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b7,r12,band_UNPAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b4,r1__Z2 <= 1
 0 <= X__b0,r11,band_UNPAIRED <= 2
 0 <= sv_function_helpervar_b4,r1__Z3 <= 1
 0 <= sv_function_helpervar_b8,r3__condX3 <= 536870910
 0 <= sv_function_helpervar_b8,r3__condX2 <= 536870910
 0 <= sv_function_helpervar_b8,r3__condX1 <= 536870910
 0 <= aux_cap_helper__b0,r11,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b0,r11,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b0,r11,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b3,r13,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b3,r13,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b8,r6,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b8,r6,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b8,r6,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b4,r5,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b4,r5,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b4,r5,band_HIGH_PAIRED__Z2 <= 1
 0 <= sv_function_helpervar_b3,r8__Z1 <= 1
 0 <= sv_function_helpervar_b3,r8__Z3 <= 1
 0 <= sv_function_helpervar_b3,r8__Z2 <= 1
 0 <= aux_cap_helper__b0,r7,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b0,r7,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b4,r11,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b9,r6,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b9,r6,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b9,r6,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b4,r11,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_b6,r2,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b1,r4,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b1,r4,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_b7,r2,band_HIGH_PAIRED <= 536870910
 0 <= X__b1,r11,band_LOW_PAIRED <= 2
 0 <= sv_function_helpervar_b4,r12__condX2 <= 536870910
 0 <= X__b9,r12,band_HIGH_PAIRED <= 3
 0 <= sv_function_helpervar_b4,r12__condX3 <= 536870910
 0 <= aux_cap_helper__b6,r5,band_UNPAIRED__condX1 <= 2
 0 <= sv_function_helpervar_b4,r12__condX1 <= 536870910
 0 <= aux_cap_helper__b6,r5,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b9,r0,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b3,r0,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b3,r0,band_UNPAIRED__condX2 <= 2
 0 <= sv_function_helpervar_b8,r12__condX2 <= 536870910
 0 <= aux_cap_helper__b9,r0,band_HIGH_PAIRED__condX1 <= 3
 0 <= sv_function_helpervar_b8,r12__condX3 <= 536870910
 0 <= aux_cap_helper__b9,r0,band_HIGH_PAIRED__condX2 <= 3
 0 <= sv_function_helpervar_b4,r1__Z1 <= 1
 0 <= sv_function_helpervar_b8,r12__condX1 <= 536870910
 0 <= aux_cap_helper__b2,r10,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b2,r10,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b2,r9,band_HIGH_PAIRED__Z2 <= 1
 0 <= X__b6,r2,band_HIGH_PAIRED <= 3
 0 <= aux_cap_helper__b2,r9,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b4,r1,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b2,r9,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b4,r1,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b4,r1,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b8,r11,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b8,r11,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b8,r11,band_HIGH_PAIRED__condX2 <= 3
 0 <= sv_function_helpervar_b2,r0__Z1 <= 1
 0 <= sv_function_helpervar_b2,r0__Z2 <= 1
 0 <= aux_cap_b3,r1,band_HIGH_PAIRED <= 536870910
 0 <= sv_function_helpervar_b2,r0__Z3 <= 1
 0 <= aux_cap_helper__b8,r1,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_b0,r8,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b8,r1,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_b3,r5,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_helper__b2,r13,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b2,r13,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b2,r13,band_HIGH_PAIRED__condX1 <= 3
 0 <= sv_function_helpervar_b3,r7__condX2 <= 536870910
 0 <= sv_function_helpervar_b3,r7__condX1 <= 536870910
 0 <= sv_function_helpervar_b3,r7__condX3 <= 536870910
 0 <= aux_cap_b3,r8,band_LOW_PAIRED <= 536870910
 0 <= sv_function_helpervar_b1,r6__condX1 <= 536870910
 0 <= aux_cap_b3,r6,band_UNPAIRED <= 536870910
 0 <= sv_function_helpervar_b1,r6__condX3 <= 536870910
 0 <= sv_function_helpervar_b1,r6__condX2 <= 536870910
 0 <= X__b6,r6,band_HIGH_PAIRED <= 3
 0 <= X__b1,r1,band_HIGH_PAIRED <= 3
 0 <= aux_cap_helper__b4,r3,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b4,r3,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b2,r0,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_b4,r8,band_UNPAIRED <= 536870910
 0 <= aux_c_b3,r8 <= 536870910
 0 <= aux_c_b3,r9 <= 536870910
 0 <= aux_c_b3,r6 <= 536870910
 0 <= aux_c_b3,r7 <= 536870910
 0 <= aux_c_b3,r4 <= 536870910
 0 <= aux_c_b3,r5 <= 536870910
 0 <= aux_cap_helper__b2,r0,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_c_b3,r2 <= 536870910
 0 <= aux_cap_helper__b0,r3,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b2,r0,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_c_b3,r3 <= 536870910
 0 <= aux_cap_helper__b0,r3,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_c_b3,r0 <= 536870910
 0 <= aux_cap_helper__b0,r3,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_c_b3,r1 <= 536870910
 0 <= aux_cap_b6,r5,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b3,r8,band_UNPAIRED__condX2 <= 2
 0 <= X__b2,r12,band_UNPAIRED <= 2
 0 <= aux_cap_b3,r11,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_helper__b3,r8,band_UNPAIRED__condX1 <= 2
 0 <= sv_function_helpervar_b2,r7__condX3 <= 536870910
 0 <= sv_function_helpervar_b2,r7__condX2 <= 536870910
 0 <= X__b0,r13,band_HIGH_PAIRED <= 3
 0 <= sv_function_helpervar_b2,r7__condX1 <= 536870910
 0 <= aux_cap_b5,r1,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_helper__b2,r8,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b2,r8,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b6,r9,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_b7,r7,band_UNPAIRED <= 536870910
 0 <= aux_sv_b1,r8 <= 536870910
 0 <= aux_cap_helper__b6,r9,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_sv_b1,r9 <= 536870910
 0 <= aux_cap_helper__b6,r9,band_HIGH_PAIRED__condX1 <= 3
 0 <= X__b4,r8,band_LOW_PAIRED <= 2
 0 <= aux_sv_b1,r6 <= 536870910
 0 <= aux_sv_b1,r7 <= 536870910
 0 <= aux_cap_helper__b1,r8,band_UNPAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b1,r0__condX2 <= 536870910
 0 <= aux_cap_helper__b1,r8,band_UNPAIRED__Z2 <= 1
 0 <= sv_function_helpervar_b1,r0__condX3 <= 536870910
 0 <= aux_cap_helper__b5,r1,band_LOW_PAIRED__condX2 <= 2
 0 <= X__b9,r11,band_LOW_PAIRED <= 2
 0 <= sv_function_helpervar_b1,r0__condX1 <= 536870910
 0 <= aux_cap_b6,r1,band_UNPAIRED <= 536870910
 0 <= aux_cap_b0,r4,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b1,r10,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b7,r10,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b7,r10,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b1,r10,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_b6,r0,band_LOW_PAIRED <= 536870910
 0 <= sv_function_helpervar_b2,r6__condX2 <= 536870910
 0 <= sv_function_helpervar_b2,r6__condX1 <= 536870910
 0 <= aux_cap_b2,r12,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b0,r0,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_b3,r2,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b5,r1,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b0,r0,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b1,r6,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b0,r0,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b1,r6,band_UNPAIRED__condX1 <= 2
 0 <= sv_function_helpervar_b2,r6__condX3 <= 536870910
 0 <= aux_cap_helper__b2,r4,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b2,r4,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b2,r4,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b1,r9,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b2,r2,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b1,r9,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b1,r9,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b9,r11,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b9,r11,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b3,r7,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b3,r7,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b3,r7,band_HIGH_PAIRED__Z2 <= 1
 0 <= sv_function_helpervar_b5,r0__condX1 <= 536870910
 0 <= sv_function_helpervar_b5,r0__condX3 <= 536870910
 0 <= sv_function_helpervar_b5,r0__condX2 <= 536870910
 0 <= sv_function_helpervar_b0,r6__Z3 <= 1
 0 <= X__b4,r12,band_LOW_PAIRED <= 2
 0 <= sv_function_helpervar_b0,r6__Z1 <= 1
 0 <= sv_function_helpervar_b0,r6__Z2 <= 1
 0 <= sv_function_helpervar_b2,r1__condX1 <= 536870910
 0 <= aux_cap_helper__b8,r9,band_HIGH_PAIRED__condX3 <= 3
 0 <= sv_function_helpervar_b2,r1__condX3 <= 536870910
 0 <= aux_cap_helper__b8,r9,band_HIGH_PAIRED__condX1 <= 3
 0 <= sv_function_helpervar_b2,r1__condX2 <= 536870910
 0 <= aux_cap_helper__b8,r9,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_b3,r10,band_HIGH_PAIRED <= 536870910
 0 <= X__b4,r10,band_HIGH_PAIRED <= 3
 0 <= aux_cap_helper__b4,r12,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b4,r12,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b8,r1,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_b8,r6,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b8,r1,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b8,r7,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b8,r7,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b2,r2,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b0,r3,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b0,r3,band_UNPAIRED__condX1 <= 2
 0 <= X__b3,r7,band_HIGH_PAIRED <= 3
 0 <= sv_function_helpervar_b2,r11__condX3 <= 536870910
 0 <= sv_function_helpervar_b2,r11__condX2 <= 536870910
 0 <= X__b2,r5,band_HIGH_PAIRED <= 3
 0 <= sv_function_helpervar_b2,r11__condX1 <= 536870910
 0 <= aux_cap_helper__b4,r13,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b5,r4,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b5,r4,band_LOW_PAIRED__condX1 <= 2
 0 <= sv_function_helpervar_b5,r11__condX1 <= 536870910
 0 <= aux_cap_helper__b9,r12,band_UNPAIRED__condX1 <= 2
 0 <= sv_function_helpervar_b5,r11__condX3 <= 536870910
 0 <= sv_function_helpervar_b5,r11__condX2 <= 536870910
 0 <= sv_function_helpervar_b6,r2__Z3 <= 1
 0 <= X__b9,r13,band_UNPAIRED <= 2
 0 <= sv_function_helpervar_b1,r7__Z2 <= 1
 0 <= sv_function_helpervar_b1,r7__Z1 <= 1
 0 <= sv_function_helpervar_b1,r7__Z3 <= 1
 0 <= aux_cap_helper__b4,r13,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_b5,r3,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_helper__b4,r13,band_HIGH_PAIRED__Z2 <= 1
 0 <= X__b2,r1,band_UNPAIRED <= 2
 0 <= aux_cap_b8,r9,band_LOW_PAIRED <= 536870910
 0 <= sv_function_helpervar_b9,r0__Z1 <= 1
 0 <= sv_function_helpervar_b9,r0__Z2 <= 1
 0 <= sv_function_helpervar_b9,r0__Z3 <= 1
 0 <= aux_cap_helper__b4,r8,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b4,r8,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b4,r8,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b5,r3,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b5,r3,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_b1,r6,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b9,r2,band_UNPAIRED__Z1 <= 1
 0 <= X__b1,r11,band_HIGH_PAIRED <= 3
 0 <= X__b3,r7,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b9,r2,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b7,r12,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b7,r12,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b7,r12,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b7,r13,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b7,r13,band_UNPAIRED__condX1 <= 2
 0 <= GlobalBidder_W_ir__b8,r10 <= 1
 0 <= aux_cap_b5,r12,band_UNPAIRED <= 536870910
 0 <= GlobalBidder_W_ir__b8,r11 <= 1
 0 <= GlobalBidder_W_ir__b8,r12 <= 1
 0 <= GlobalBidder_W_ir__b8,r13 <= 1
 0 <= aux_cap_helper__b1,r12,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b1,r12,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b1,r12,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b4,r11,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b4,r11,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b9,r12,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b4,r11,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b7,r13,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b7,r13,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_b5,r7,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b7,r13,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b0,r9,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b0,r9,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b0,r9,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b1,r0,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b6,r7,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b6,r7,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_b3,r3,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b6,r7,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_c_b0,r7 <= 536870910
 0 <= aux_c_b0,r8 <= 536870910
 0 <= aux_c_b0,r9 <= 536870910
 0 <= aux_cap_helper__b2,r1,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b3,r11,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_b1,r11,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b2,r1,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b3,r11,band_UNPAIRED__condX2 <= 2
 0 <= aux_c_b0,r3 <= 536870910
 0 <= aux_cap_helper__b1,r0,band_HIGH_PAIRED__condX2 <= 3
 0 <= sv_function_helpervar_b7,r12__Z3 <= 1
 0 <= aux_c_b0,r4 <= 536870910
 0 <= aux_cap_helper__b1,r0,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_c_b0,r5 <= 536870910
 0 <= aux_cap_helper__b2,r10,band_LOW_PAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b7,r12__Z1 <= 1
 0 <= aux_c_b0,r6 <= 536870910
 0 <= aux_cap_helper__b2,r10,band_LOW_PAIRED__Z2 <= 1
 0 <= sv_function_helpervar_b7,r12__Z2 <= 1
 0 <= aux_c_b0,r0 <= 536870910
 0 <= aux_c_b0,r1 <= 536870910
 0 <= aux_c_b0,r2 <= 536870910
 0 <= aux_Omega_b5,r13 <= 536870910
 0 <= aux_cap_helper__b8,r11,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b8,r11,band_UNPAIRED__Z1 <= 1
 0 <= X__b0,r8,band_UNPAIRED <= 2
 0 <= aux_Omega_b5,r10 <= 536870910
 0 <= aux_cap_helper__b0,r12,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_Omega_b5,r11 <= 536870910
 0 <= aux_cap_helper__b0,r12,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_Omega_b5,r12 <= 536870910
 0 <= aux_sv_b4,r0 <= 536870910
 0 <= aux_sv_b4,r8 <= 536870910
 0 <= aux_sv_b4,r7 <= 536870910
 0 <= aux_sv_b4,r6 <= 536870910
 0 <= X__b4,r5,band_UNPAIRED <= 2
 0 <= aux_sv_b4,r5 <= 536870910
 0 <= aux_sv_b4,r4 <= 536870910
 0 <= aux_sv_b4,r3 <= 536870910
 0 <= aux_sv_b4,r2 <= 536870910
 0 <= aux_cap_b1,r8,band_LOW_PAIRED <= 536870910
 0 <= aux_sv_b4,r1 <= 536870910
 0 <= aux_cap_helper__b2,r11,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b2,r11,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b0,r6,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b0,r6,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_sv_b4,r9 <= 536870910
 0 <= X__b1,r13,band_HIGH_PAIRED <= 3
 0 <= X__b3,r7,band_LOW_PAIRED <= 2
 0 <= X__b6,r4,band_LOW_PAIRED <= 2
 0 <= aux_cap_helper__b2,r9,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b2,r9,band_LOW_PAIRED__condX2 <= 2
 0 <= X__b7,r12,band_UNPAIRED <= 2
 0 <= sv_function_helpervar_b6,r5__condX1 <= 536870910
 0 <= sv_function_helpervar_b6,r5__condX3 <= 536870910
 0 <= sv_function_helpervar_b6,r5__condX2 <= 536870910
 0 <= aux_cap_helper__b2,r5,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b2,r5,band_UNPAIRED__condX1 <= 2
 0 <= sv_function_helpervar_b2,r10__Z3 <= 1
 0 <= aux_cap_helper__b5,r10,band_UNPAIRED__condX2 <= 2
 0 <= sv_function_helpervar_b2,r10__Z2 <= 1
 0 <= sv_function_helpervar_b2,r10__Z1 <= 1
 0 <= aux_cap_helper__b5,r10,band_UNPAIRED__condX1 <= 2
 0 <= X__b1,r6,band_UNPAIRED <= 2
 0 <= aux_cap_b6,r9,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b7,r3,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_c_b2,r11 <= 536870910
 0 <= aux_c_b2,r10 <= 536870910
 0 <= X__b0,r6,band_HIGH_PAIRED <= 3
 0 <= aux_c_b2,r13 <= 536870910
 0 <= aux_c_b2,r12 <= 536870910
 0 <= aux_cap_helper__b0,r9,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b0,r9,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b5,r6,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b5,r6,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b5,r6,band_HIGH_PAIRED__Z3 <= 1
 0 <= sv_function_helpervar_b5,r4__condX3 <= 536870910
 0 <= sv_function_helpervar_b5,r4__condX2 <= 536870910
 0 <= sv_function_helpervar_b5,r4__condX1 <= 536870910
 0 <= X__b3,r13,band_HIGH_PAIRED <= 3
 0 <= aux_cap_b4,r5,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b3,r11,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b3,r11,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b7,r3,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b9,r1,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b9,r1,band_LOW_PAIRED__condX1 <= 2
 0 <= sv_function_helpervar_b3,r4__Z1 <= 1
 0 <= sv_function_helpervar_b3,r4__Z3 <= 1
 0 <= sv_function_helpervar_b3,r4__Z2 <= 1
 0 <= aux_Omega_b6,r3 <= 536870910
 0 <= aux_Omega_b6,r4 <= 536870910
 0 <= aux_cap_b2,r7,band_LOW_PAIRED <= 536870910
 0 <= aux_Omega_b6,r5 <= 536870910
 0 <= aux_Omega_b6,r6 <= 536870910
 0 <= sv_function_helpervar_b5,r9__Z1 <= 1
 0 <= aux_Omega_b6,r7 <= 536870910
 0 <= sv_function_helpervar_b5,r9__Z2 <= 1
 0 <= aux_Omega_b6,r8 <= 536870910
 0 <= aux_Omega_b6,r9 <= 536870910
 0 <= aux_cap_helper__b5,r8,band_HIGH_PAIRED__condX3 <= 3
 0 <= sv_function_helpervar_b5,r9__Z3 <= 1
 0 <= aux_cap_b4,r10,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_helper__b5,r8,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_b2,r0,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b5,r8,band_HIGH_PAIRED__condX1 <= 3
 0 <= X__b8,r1,band_LOW_PAIRED <= 2
 0 <= aux_cap_b0,r4,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b3,r4,band_UNPAIRED__condX1 <= 2
 0 <= X__b4,r7,band_HIGH_PAIRED <= 3
 0 <= aux_cap_helper__b3,r4,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b3,r10,band_LOW_PAIRED__Z2 <= 1
 0 <= X__b3,r5,band_HIGH_PAIRED <= 3
 0 <= X__b0,r1,band_LOW_PAIRED <= 2
 0 <= aux_cap_helper__b3,r10,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b6,r5,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b6,r5,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_b1,r6,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b8,r8,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b8,r8,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_b5,r3,band_UNPAIRED <= 536870910
 0 <= aux_Omega_b6,r0 <= 536870910
 0 <= aux_Omega_b6,r1 <= 536870910
 0 <= aux_Omega_b6,r2 <= 536870910
 0 <= aux_cap_helper__b7,r5,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b7,r5,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b7,r5,band_HIGH_PAIRED__Z2 <= 1
 0 <= X__b6,r10,band_HIGH_PAIRED <= 3
 0 <= aux_cap_helper__b5,r5,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_Omega_b9,r13 <= 536870910
 0 <= aux_cap_helper__b5,r5,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b5,r5,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b3,r12,band_UNPAIRED__Z1 <= 1
 0 <= aux_Omega_b9,r10 <= 536870910
 0 <= aux_cap_helper__b3,r12,band_UNPAIRED__Z2 <= 1
 0 <= aux_Omega_b9,r11 <= 536870910
 0 <= aux_Omega_b9,r12 <= 536870910
 0 <= aux_cap_helper__b3,r10,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b3,r10,band_UNPAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b8,r7__Z1 <= 1
 0 <= sv_function_helpervar_b8,r7__Z3 <= 1
 0 <= sv_function_helpervar_b8,r7__Z2 <= 1
 0 <= X__b8,r1,band_HIGH_PAIRED <= 3
 0 <= X__b4,r6,band_LOW_PAIRED <= 2
 0 <= X__b4,r12,band_UNPAIRED <= 2
 0 <= sv_function_helpervar_b0,r0__condX3 <= 536870910
 0 <= sv_function_helpervar_b0,r0__condX2 <= 536870910
 0 <= sv_function_helpervar_b0,r0__condX1 <= 536870910
 0 <= aux_cap_helper__b9,r2,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b9,r2,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b9,r5,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b3,r4,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b3,r4,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b9,r5,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b3,r4,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_b8,r12,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b1,r11,band_LOW_PAIRED__Z2 <= 1
 0 <= X__b5,r12,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b1,r11,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b1,r12,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b7,r2,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b7,r2,band_LOW_PAIRED__condX1 <= 2
 0 <= X__b0,r0,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b4,r8,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b4,r8,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b1,r12,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b4,r8,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b1,r8,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b9,r0,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b1,r8,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b9,r0,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b0,r9,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b0,r9,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_b5,r12,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_b2,r4,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b1,r2,band_UNPAIRED__condX2 <= 2
 0 <= GlobalBidder_hat{W}__i9,k2 <= 1
 0 <= GlobalBidder_W_ir__b9,r2 <= 1
 0 <= aux_cap_helper__b1,r2,band_UNPAIRED__condX1 <= 2
 0 <= GlobalBidder_hat{W}__i9,k1 <= 1
 0 <= GlobalBidder_W_ir__b9,r3 <= 1
 0 <= GlobalBidder_W_ir__b9,r4 <= 1
 0 <= GlobalBidder_W_ir__b9,r5 <= 1
 0 <= GlobalBidder_hat{W}__i9,k0 <= 1
 0 <= GlobalBidder_W_ir__b9,r0 <= 1
 0 <= GlobalBidder_W_ir__b9,r1 <= 1
 0 <= aux_cap_helper__b1,r12,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b1,r12,band_HIGH_PAIRED__condX2 <= 3
 0 <= sv_function_helpervar_b1,r12__condX2 <= 536870910
 0 <= aux_cap_helper__b7,r9,band_HIGH_PAIRED__condX3 <= 3
 0 <= sv_function_helpervar_b1,r12__condX1 <= 536870910
 0 <= aux_cap_helper__b7,r9,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b7,r9,band_HIGH_PAIRED__condX1 <= 3
 0 <= sv_function_helpervar_b1,r12__condX3 <= 536870910
 0 <= X__b7,r2,band_LOW_PAIRED <= 2
 0 <= X__b5,r5,band_LOW_PAIRED <= 2
 0 <= aux_cap_helper__b2,r3,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b2,r3,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b1,r12,band_HIGH_PAIRED__condX1 <= 3
 0 <= sv_function_helpervar_b6,r2__Z1 <= 1
 0 <= sv_function_helpervar_b6,r2__Z2 <= 1
 0 <= aux_cap_helper__b1,r3,band_LOW_PAIRED__condX2 <= 2
 0 <= X__b3,r12,band_LOW_PAIRED <= 2
 0 <= aux_cap_helper__b1,r3,band_LOW_PAIRED__condX1 <= 2
 0 <= sv_function_helpervar_b9,r3__condX2 <= 536870910
 0 <= sv_function_helpervar_b9,r3__condX1 <= 536870910
 0 <= aux_cap_helper__b4,r7,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_b2,r1,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b4,r7,band_UNPAIRED__condX2 <= 2
 0 <= sv_function_helpervar_b9,r3__condX3 <= 536870910
 0 <= sv_function_helpervar_b1,r3__Z2 <= 1
 0 <= sv_function_helpervar_b1,r3__Z1 <= 1
 0 <= sv_function_helpervar_b1,r3__Z3 <= 1
 0 <= aux_cap_helper__b2,r10,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b2,r10,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b8,r13,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b8,r13,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b6,r9,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b6,r9,band_LOW_PAIRED__condX2 <= 2
 0 <= X__b9,r3,band_HIGH_PAIRED <= 3
 0 <= aux_cap_helper__b2,r13,band_LOW_PAIRED__condX2 <= 2
 0 <= X__b2,r5,band_LOW_PAIRED <= 2
 0 <= aux_cap_helper__b4,r2,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b4,r2,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b4,r2,band_HIGH_PAIRED__Z2 <= 1
 0 <= sv_function_helpervar_b4,r9__Z1 <= 1
 0 <= aux_cap_helper__b2,r13,band_LOW_PAIRED__condX1 <= 2
 0 <= sv_function_helpervar_b2,r8__condX1 <= 536870910
 0 <= sv_function_helpervar_b4,r9__Z3 <= 1
 0 <= sv_function_helpervar_b2,r8__condX2 <= 536870910
 0 <= sv_function_helpervar_b4,r9__Z2 <= 1
 0 <= sv_function_helpervar_b2,r8__condX3 <= 536870910
 0 <= aux_cap_helper__b3,r3,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b3,r3,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_b5,r6,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b8,r6,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_b7,r7,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b8,r6,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b8,r6,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b5,r11,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b5,r11,band_LOW_PAIRED__condX1 <= 2
 0 <= X__b8,r8,band_HIGH_PAIRED <= 3
 0 <= aux_cap_b4,r0,band_UNPAIRED <= 536870910
 0 <= sv_function_helpervar_b9,r8__condX3 <= 536870910
 0 <= sv_function_helpervar_b9,r8__condX2 <= 536870910
 0 <= sv_function_helpervar_b9,r8__condX1 <= 536870910
 0 <= aux_cap_helper__b7,r0,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b7,r0,band_LOW_PAIRED__condX2 <= 2
 0 <= X__b3,r3,band_HIGH_PAIRED <= 3
 0 <= X__b3,r13,band_LOW_PAIRED <= 2
 0 <= aux_cap_b4,r9,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b7,r2,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b9,r12,band_HIGH_PAIRED__Z2 <= 1
 0 <= sv_function_helpervar_b6,r11__Z3 <= 1
 0 <= aux_cap_helper__b9,r12,band_HIGH_PAIRED__Z3 <= 1
 0 <= sv_function_helpervar_b6,r11__Z2 <= 1
 0 <= sv_function_helpervar_b6,r11__Z1 <= 1
 0 <= aux_cap_helper__b9,r12,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b7,r2,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b7,r2,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b9,r4,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b9,r4,band_LOW_PAIRED__condX1 <= 2
 0 <= X__b0,r5,band_HIGH_PAIRED <= 3
 0 <= aux_cap_helper__b7,r13,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b7,r13,band_UNPAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b9,r11__Z1 <= 1
 0 <= sv_function_helpervar_b9,r11__Z2 <= 1
 0 <= sv_function_helpervar_b9,r11__Z3 <= 1
 0 <= aux_cap_helper__b9,r3,band_HIGH_PAIRED__Z1 <= 1
 0 <= X__b0,r0,band_LOW_PAIRED <= 2
 0 <= aux_cap_helper__b9,r3,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b9,r3,band_HIGH_PAIRED__Z3 <= 1
 0 <= sv_function_helpervar_b5,r4__Z3 <= 1
 0 <= sv_function_helpervar_b5,r4__Z2 <= 1
 0 <= sv_function_helpervar_b5,r4__Z1 <= 1
 0 <= X__b6,r3,band_LOW_PAIRED <= 2
 0 <= aux_cap_helper__b3,r10,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b6,r1,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b9,r4,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b6,r1,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b9,r4,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b3,r10,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b3,r10,band_HIGH_PAIRED__Z3 <= 1
 0 <= X__b9,r8,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b8,r10,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b6,r13,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b8,r10,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b6,r13,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b6,r12,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b7,r7,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b6,r12,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_b1,r5,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_b5,r5,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_helper__b6,r12,band_HIGH_PAIRED__condX3 <= 3
 0 <= sv_function_helpervar_b0,r3__Z3 <= 1
 0 <= sv_function_helpervar_b0,r3__Z2 <= 1
 0 <= sv_function_helpervar_b0,r3__Z1 <= 1
 0 <= sv_function_helpervar_b9,r2__condX2 <= 536870910
 0 <= sv_function_helpervar_b9,r2__condX3 <= 536870910
 0 <= sv_function_helpervar_b5,r13__condX1 <= 536870910
 0 <= X__b4,r10,band_LOW_PAIRED <= 2
 0 <= sv_function_helpervar_b5,r13__condX3 <= 536870910
 0 <= sv_function_helpervar_b5,r13__condX2 <= 536870910
 0 <= aux_cap_helper__b5,r10,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b5,r10,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_Omega_b1,r13 <= 536870910
 0 <= aux_cap_helper__b5,r0,band_UNPAIRED__condX2 <= 2
 0 <= aux_Omega_b1,r10 <= 536870910
 0 <= aux_cap_helper__b5,r0,band_UNPAIRED__condX1 <= 2
 0 <= aux_Omega_b1,r11 <= 536870910
 0 <= aux_Omega_b1,r12 <= 536870910
 0 <= aux_cap_b6,r5,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_helper__b7,r7,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b0,r9,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b0,r9,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_b5,r10,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_b0,r11,band_LOW_PAIRED <= 536870910
 0 <= aux_Omega_b0,r9 <= 536870910
 0 <= aux_Omega_b0,r6 <= 536870910
 0 <= aux_cap_helper__b8,r1,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_Omega_b0,r5 <= 536870910
 0 <= aux_cap_helper__b8,r1,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_Omega_b0,r8 <= 536870910
 0 <= aux_Omega_b0,r7 <= 536870910
 0 <= aux_cap_helper__b8,r1,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_Omega_b0,r2 <= 536870910
 0 <= aux_Omega_b0,r1 <= 536870910
 0 <= aux_Omega_b0,r4 <= 536870910
 0 <= aux_cap_helper__b9,r7,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_Omega_b0,r3 <= 536870910
 0 <= aux_cap_helper__b9,r7,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b9,r7,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_Omega_b0,r0 <= 536870910
 0 <= sv_function_helpervar_b7,r9__Z1 <= 1
 0 <= aux_cap_b4,r3,band_HIGH_PAIRED <= 536870910
 0 <= sv_function_helpervar_b7,r9__Z2 <= 1
 0 <= sv_function_helpervar_b7,r9__Z3 <= 1
 0 <= X__b2,r9,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b2,r2,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b2,r2,band_UNPAIRED__Z2 <= 1
 0 <= X__b1,r12,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b1,r8,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b1,r8,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b7,r3,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b7,r3,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b7,r3,band_HIGH_PAIRED__condX1 <= 3
 0 <= X__b1,r3,band_UNPAIRED <= 2
 0 <= X__b2,r4,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b7,r7,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b7,r7,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b7,r7,band_HIGH_PAIRED__Z1 <= 1
 0 <= X__b5,r4,band_HIGH_PAIRED <= 3
 0 <= aux_cap_helper__b0,r0,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b0,r0,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b0,r0,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b0,r3,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b0,r3,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b0,r3,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b7,r3,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_b4,r0,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_helper__b7,r3,band_UNPAIRED__condX2 <= 2
 0 <= sv_function_helpervar_b1,r9__Z3 <= 1
 0 <= aux_cap_b6,r3,band_HIGH_PAIRED <= 536870910
 0 <= sv_function_helpervar_b1,r9__Z2 <= 1
 0 <= sv_function_helpervar_b1,r9__Z1 <= 1
 0 <= X__b8,r0,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b0,r7,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b1,r1,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b1,r1,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b0,r7,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b0,r7,band_HIGH_PAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b1,r10__condX1 <= 536870910
 0 <= sv_function_helpervar_b1,r10__condX2 <= 536870910
 0 <= sv_function_helpervar_b1,r10__condX3 <= 536870910
 0 <= aux_cap_helper__b6,r13,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b6,r13,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b6,r13,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b6,r13,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b6,r13,band_UNPAIRED__condX2 <= 2
 0 <= X__b8,r1,band_UNPAIRED <= 2
 0 <= X__b2,r3,band_UNPAIRED <= 2
 0 <= sv_function_helpervar_b7,r9__condX1 <= 536870910
 0 <= sv_function_helpervar_b7,r9__condX2 <= 536870910
 0 <= sv_function_helpervar_b7,r9__condX3 <= 536870910
 0 <= aux_cap_helper__b0,r0,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b0,r0,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b1,r12,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b1,r12,band_LOW_PAIRED__Z2 <= 1
 0 <= sv_function_helpervar_b2,r8__Z1 <= 1
 0 <= sv_function_helpervar_b2,r8__Z2 <= 1
 0 <= X__b8,r12,band_LOW_PAIRED <= 2
 0 <= sv_function_helpervar_b2,r8__Z3 <= 1
 0 <= sv_function_helpervar_b8,r0__Z3 <= 1
 0 <= sv_function_helpervar_b8,r0__Z2 <= 1
 0 <= sv_function_helpervar_b8,r0__Z1 <= 1
 0 <= aux_cap_b9,r8,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b5,r13,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b5,r13,band_UNPAIRED__Z2 <= 1
 0 <= X__b8,r2,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b2,r8,band_HIGH_PAIRED__condX1 <= 3
 0 <= X__b8,r8,band_LOW_PAIRED <= 2
 0 <= aux_cap_helper__b4,r1,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b4,r1,band_LOW_PAIRED__Z1 <= 1
 0 <= X__b8,r2,band_LOW_PAIRED <= 2
 0 <= aux_cap_helper__b1,r7,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b1,r7,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b6,r0,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b5,r6,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b6,r0,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b2,r8,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b5,r6,band_HIGH_PAIRED__condX2 <= 3
 0 <= X__b8,r7,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b2,r8,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b5,r6,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_b2,r0,band_LOW_PAIRED <= 536870910
 0 <= aux_c_b4,r13 <= 536870910
 0 <= aux_c_b4,r12 <= 536870910
 0 <= aux_cap_helper__b4,r4,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_b2,r6,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_helper__b4,r4,band_LOW_PAIRED__condX2 <= 2
 0 <= X__b1,r13,band_UNPAIRED <= 2
 0 <= aux_c_b4,r11 <= 536870910
 0 <= aux_c_b4,r10 <= 536870910
 0 <= aux_cap_helper__b9,r13,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b9,r13,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b6,r12,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b6,r12,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b4,r4,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b4,r4,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b4,r4,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b4,r6,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b4,r6,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b4,r6,band_HIGH_PAIRED__Z3 <= 1
 0 <= sv_function_helpervar_b0,r9__condX1 <= 536870910
 0 <= aux_cap_b1,r4,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b4,r0,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b4,r0,band_UNPAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b0,r9__condX3 <= 536870910
 0 <= aux_c_b4,r9 <= 536870910
 0 <= sv_function_helpervar_b0,r9__condX2 <= 536870910
 0 <= aux_sv_b3,r10 <= 536870910
 0 <= aux_c_b4,r7 <= 536870910
 0 <= aux_cap_helper__b4,r12,band_UNPAIRED__condX1 <= 2
 0 <= aux_c_b4,r8 <= 536870910
 0 <= aux_cap_helper__b4,r12,band_UNPAIRED__condX2 <= 2
 0 <= aux_sv_b3,r13 <= 536870910
 0 <= aux_c_b4,r5 <= 536870910
 0 <= aux_c_b4,r6 <= 536870910
 0 <= aux_sv_b3,r11 <= 536870910
 0 <= aux_c_b4,r3 <= 536870910
 0 <= aux_sv_b3,r12 <= 536870910
 0 <= aux_c_b4,r4 <= 536870910
 0 <= aux_c_b4,r1 <= 536870910
 0 <= aux_c_b4,r2 <= 536870910
 0 <= aux_c_b4,r0 <= 536870910
 0 <= aux_cap_b6,r11,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b4,r5,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_b9,r2,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b4,r5,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b4,r5,band_HIGH_PAIRED__condX2 <= 3
 0 <= sv_function_helpervar_b5,r10__Z1 <= 1
 0 <= sv_function_helpervar_b5,r10__Z2 <= 1
 0 <= sv_function_helpervar_b5,r10__Z3 <= 1
 0 <= aux_c_b1,r9 <= 536870910
 0 <= sv_function_helpervar_b1,r2__condX1 <= 536870910
 0 <= sv_function_helpervar_b4,r11__condX2 <= 536870910
 0 <= aux_c_b1,r8 <= 536870910
 0 <= sv_function_helpervar_b4,r11__condX1 <= 536870910
 0 <= sv_function_helpervar_b4,r11__condX3 <= 536870910
 0 <= aux_c_b1,r5 <= 536870910
 0 <= aux_c_b1,r4 <= 536870910
 0 <= aux_c_b1,r7 <= 536870910
 0 <= aux_Omega_b3,r8 <= 536870910
 0 <= aux_c_b1,r6 <= 536870910
 0 <= sv_function_helpervar_b1,r12__Z3 <= 1
 0 <= aux_Omega_b3,r9 <= 536870910
 0 <= aux_c_b1,r1 <= 536870910
 0 <= aux_Omega_b3,r6 <= 536870910
 0 <= aux_cap_helper__b7,r4,band_UNPAIRED__condX1 <= 2
 0 <= aux_c_b1,r0 <= 536870910
 0 <= aux_Omega_b3,r7 <= 536870910
 0 <= aux_cap_helper__b7,r4,band_UNPAIRED__condX2 <= 2
 0 <= aux_c_b1,r3 <= 536870910
 0 <= aux_Omega_b3,r4 <= 536870910
 0 <= aux_c_b1,r2 <= 536870910
 0 <= aux_Omega_b3,r5 <= 536870910
 0 <= aux_Omega_b3,r2 <= 536870910
 0 <= aux_Omega_b3,r3 <= 536870910
 0 <= aux_cap_helper__b1,r2,band_LOW_PAIRED__Z2 <= 1
 0 <= sv_function_helpervar_b1,r2__condX3 <= 536870910
 0 <= aux_Omega_b3,r0 <= 536870910
 0 <= aux_cap_helper__b1,r2,band_LOW_PAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b1,r2__condX2 <= 536870910
 0 <= aux_Omega_b3,r1 <= 536870910
 0 <= aux_cap_helper__b0,r0,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b4,r6,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b0,r0,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b4,r6,band_LOW_PAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b1,r12__Z2 <= 1
 0 <= sv_function_helpervar_b1,r12__Z1 <= 1
 0 <= aux_cap_b5,r0,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_helper__b8,r4,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b8,r4,band_UNPAIRED__Z1 <= 1
 0 <= X__b2,r0,band_HIGH_PAIRED <= 3
 0 <= X__b3,r8,band_UNPAIRED <= 2
 0 <= sv_function_helpervar_b5,r0__Z3 <= 1
 0 <= sv_function_helpervar_b5,r0__Z2 <= 1
 0 <= sv_function_helpervar_b5,r0__Z1 <= 1
 0 <= aux_cap_helper__b9,r6,band_UNPAIRED__Z2 <= 1
 0 <= X__b5,r13,band_HIGH_PAIRED <= 3
 0 <= aux_cap_helper__b9,r6,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b0,r12,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b0,r12,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b0,r12,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_b0,r13,band_HIGH_PAIRED <= 536870910
 0 <= sv_function_helpervar_b4,r5__Z1 <= 1
 0 <= X__b2,r2,band_UNPAIRED <= 2
 0 <= sv_function_helpervar_b4,r5__Z2 <= 1
 0 <= sv_function_helpervar_b4,r5__Z3 <= 1
 0 <= aux_cap_helper__b7,r5,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b7,r5,band_LOW_PAIRED__Z1 <= 1
 0 <= X__b8,r9,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b4,r8,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b4,r8,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b4,r11,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b4,r11,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b7,r10,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_b6,r4,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b7,r10,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b5,r9,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b5,r9,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b5,r9,band_HIGH_PAIRED__condX1 <= 3
 0 <= X__b1,r13,band_LOW_PAIRED <= 2
 0 <= X__b5,r3,band_LOW_PAIRED <= 2
 0 <= aux_cap_b3,r12,band_UNPAIRED <= 536870910
 0 <= aux_cap_b2,r9,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b8,r4,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b8,r4,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b8,r4,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_b2,r11,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_b7,r13,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b6,r1,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b6,r1,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_b4,r6,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b6,r2,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b0,r2,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_b3,r0,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b6,r2,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b0,r2,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b6,r1,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b0,r2,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b9,r3,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b9,r3,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b3,r5,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b3,r5,band_LOW_PAIRED__condX1 <= 2
 0 <= X__b9,r6,band_UNPAIRED <= 2
 0 <= X__b7,r1,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b9,r10,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b9,r10,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_b2,r10,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b9,r6,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b9,r6,band_LOW_PAIRED__condX2 <= 2
 0 <= X__b6,r8,band_HIGH_PAIRED <= 3
 0 <= sv_function_helpervar_b3,r4__condX1 <= 536870910
 0 <= sv_function_helpervar_b3,r4__condX3 <= 536870910
 0 <= sv_function_helpervar_b3,r4__condX2 <= 536870910
 0 <= aux_cap_b3,r5,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_b0,r0,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b1,r6,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b1,r6,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_b0,r2,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_b0,r8,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_helper__b7,r4,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b7,r4,band_UNPAIRED__Z1 <= 1
 0 <= X__b1,r5,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b0,r12,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b0,r12,band_UNPAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b3,r13__Z2 <= 1
 0 <= sv_function_helpervar_b3,r13__Z3 <= 1
 0 <= sv_function_helpervar_b3,r13__Z1 <= 1
 0 <= aux_cap_helper__b5,r13,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b5,r13,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b2,r9,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b2,r9,band_UNPAIRED__condX1 <= 2
 0 <= X__b7,r3,band_UNPAIRED <= 2
 0 <= aux_cap_b4,r8,band_HIGH_PAIRED <= 536870910
 0 <= aux_sv_b7,r10 <= 536870910
 0 <= aux_cap_helper__b8,r0,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b8,r0,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b1,r7,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b1,r7,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_sv_b7,r13 <= 536870910
 0 <= aux_sv_b7,r12 <= 536870910
 0 <= aux_sv_b7,r11 <= 536870910
 0 <= aux_cap_b4,r13,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_helper__b1,r7,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b4,r6,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b4,r6,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b7,r2,band_LOW_PAIRED__Z1 <= 1
 0 <= X__b3,r10,band_HIGH_PAIRED <= 3
 0 <= aux_cap_helper__b3,r12,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b7,r2,band_LOW_PAIRED__Z2 <= 1
 0 <= X__b4,r11,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b0,r4,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b3,r12,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b0,r4,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b2,r7,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b2,r7,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b7,r13,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b2,r7,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b7,r13,band_LOW_PAIRED__Z2 <= 1
 0 <= X__b9,r2,band_LOW_PAIRED <= 2
 0 <= aux_cap_b1,r0,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_b7,r13,band_UNPAIRED <= 536870910
 0 <= sv_function_helpervar_b8,r12__Z2 <= 1
 0 <= sv_function_helpervar_b8,r12__Z3 <= 1
 0 <= sv_function_helpervar_b8,r12__Z1 <= 1
 0 <= aux_cap_helper__b3,r13,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b7,r8,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_b1,r6,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_helper__b3,r13,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b7,r8,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b3,r13,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b7,r8,band_HIGH_PAIRED__Z3 <= 1
 0 <= X__b2,r10,band_HIGH_PAIRED <= 3
 0 <= sv_function_helpervar_b2,r4__Z1 <= 1
 0 <= sv_function_helpervar_b2,r4__Z2 <= 1
 0 <= sv_function_helpervar_b2,r4__Z3 <= 1
 0 <= aux_cap_helper__b0,r2,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b0,r2,band_HIGH_PAIRED__Z3 <= 1
 0 <= sv_function_helpervar_b7,r5__Z1 <= 1
 0 <= aux_cap_helper__b7,r0,band_HIGH_PAIRED__condX3 <= 3
 0 <= sv_function_helpervar_b7,r5__Z2 <= 1
 0 <= aux_cap_helper__b7,r0,band_HIGH_PAIRED__condX2 <= 3
 0 <= sv_function_helpervar_b7,r5__Z3 <= 1
 0 <= aux_cap_helper__b0,r2,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b7,r0,band_HIGH_PAIRED__condX1 <= 3
 0 <= sv_function_helpervar_b9,r2__condX1 <= 536870910
 0 <= sv_function_helpervar_b9,r6__Z2 <= 1
 0 <= sv_function_helpervar_b9,r6__Z3 <= 1
 0 <= aux_cap_helper__b6,r10,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b6,r10,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_c_b8,r11 <= 536870910
 0 <= aux_cap_helper__b6,r10,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_c_b8,r10 <= 536870910
 0 <= sv_function_helpervar_b9,r6__Z1 <= 1
 0 <= aux_c_b8,r13 <= 536870910
 0 <= aux_c_b8,r12 <= 536870910
 0 <= aux_cap_helper__b1,r4,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b1,r4,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b1,r4,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b1,r6,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_sv_b9,r8 <= 536870910
 0 <= aux_cap_helper__b1,r6,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_sv_b9,r9 <= 536870910
 0 <= aux_sv_b9,r6 <= 536870910
 0 <= X__b3,r9,band_HIGH_PAIRED <= 3
 0 <= aux_sv_b9,r7 <= 536870910
 0 <= X__b6,r7,band_HIGH_PAIRED <= 3
 0 <= aux_cap_helper__b2,r6,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_sv_b9,r0 <= 536870910
 0 <= aux_cap_helper__b0,r11,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b2,r6,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_sv_b9,r1 <= 536870910
 0 <= aux_cap_helper__b0,r11,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b2,r6,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b0,r11,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_sv_b9,r4 <= 536870910
 0 <= aux_sv_b9,r5 <= 536870910
 0 <= X__b4,r3,band_LOW_PAIRED <= 2
 0 <= aux_sv_b9,r2 <= 536870910
 0 <= aux_sv_b9,r3 <= 536870910
 0 <= aux_cap_b4,r7,band_LOW_PAIRED <= 536870910
 0 <= sv_function_helpervar_b7,r0__condX1 <= 536870910
 0 <= aux_cap_helper__b3,r0,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b3,r0,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b3,r0,band_HIGH_PAIRED__condX1 <= 3
 0 <= GlobalBidder_W_ir__b8,r8 <= 1
 0 <= GlobalBidder_W_ir__b8,r7 <= 1
 0 <= GlobalBidder_W_ir__b8,r6 <= 1
 0 <= aux_cap_b8,r13,band_HIGH_PAIRED <= 536870910
 0 <= GlobalBidder_W_ir__b8,r5 <= 1
 0 <= GlobalBidder_W_ir__b8,r9 <= 1
 0 <= aux_cap_helper__b9,r6,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b6,r13,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b6,r13,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b9,r6,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b6,r13,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b9,r6,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b6,r3,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b6,r2,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b6,r3,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b6,r2,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b6,r2,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b7,r5,band_HIGH_PAIRED__condX3 <= 3
 0 <= GlobalBidder_W_ir__b8,r0 <= 1
 0 <= aux_cap_helper__b9,r13,band_LOW_PAIRED__condX2 <= 2
 0 <= X__b0,r4,band_HIGH_PAIRED <= 3
 0 <= aux_cap_b7,r3,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_helper__b7,r5,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b9,r13,band_LOW_PAIRED__condX1 <= 2
 0 <= GlobalBidder_W_ir__b8,r4 <= 1
 0 <= X__b7,r3,band_HIGH_PAIRED <= 3
 0 <= GlobalBidder_W_ir__b8,r3 <= 1
 0 <= aux_cap_b7,r9,band_LOW_PAIRED <= 536870910
 0 <= GlobalBidder_W_ir__b8,r2 <= 1
 0 <= GlobalBidder_W_ir__b8,r1 <= 1
 0 <= aux_cap_helper__b7,r5,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b3,r8,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b3,r8,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b2,r6,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b9,r1,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b9,r1,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_b0,r1,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b2,r6,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b6,r7,band_LOW_PAIRED__Z1 <= 1
 0 <= X__b3,r5,band_LOW_PAIRED <= 2
 0 <= aux_cap_helper__b6,r7,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b2,r12,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b4,r2,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b2,r12,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b4,r2,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b4,r2,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_b5,r4,band_UNPAIRED <= 536870910
 0 <= X__b9,r8,band_LOW_PAIRED <= 2
 0 <= aux_cap_helper__b5,r11,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b5,r11,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b5,r11,band_HIGH_PAIRED__condX1 <= 3
 0 <= sv_function_helpervar_b0,r6__condX3 <= 536870910
 0 <= sv_function_helpervar_b0,r6__condX2 <= 536870910
 0 <= sv_function_helpervar_b0,r6__condX1 <= 536870910
 0 <= X__b9,r6,band_HIGH_PAIRED <= 3
 0 <= aux_cap_helper__b8,r7,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b8,r7,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b8,r7,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b8,r7,band_UNPAIRED__condX1 <= 2
 0 <= X__b9,r0,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b8,r7,band_UNPAIRED__condX2 <= 2
 0 <= sv_function_helpervar_b7,r0__condX2 <= 536870910
 0 <= sv_function_helpervar_b7,r0__condX3 <= 536870910
 0 <= X__b5,r13,band_UNPAIRED <= 2
 0 <= aux_cap_b7,r12,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b1,r12,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b1,r12,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b9,r10,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b9,r10,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b4,r10,band_LOW_PAIRED__condX2 <= 2
 0 <= X__b1,r7,band_LOW_PAIRED <= 2
 0 <= aux_cap_helper__b4,r10,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b6,r12,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b6,r12,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b3,r9,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b4,r0,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b3,r9,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b3,r9,band_HIGH_PAIRED__condX1 <= 3
 0 <= sv_function_helpervar_b4,r2__Z3 <= 1
 0 <= sv_function_helpervar_b4,r2__Z1 <= 1
 0 <= sv_function_helpervar_b4,r2__Z2 <= 1
 0 <= sv_function_helpervar_b3,r7__Z2 <= 1
 0 <= sv_function_helpervar_b3,r7__Z1 <= 1
 0 <= sv_function_helpervar_b2,r13__Z2 <= 1
 0 <= aux_Omega_b8,r12 <= 536870910
 0 <= sv_function_helpervar_b2,r13__Z1 <= 1
 0 <= aux_Omega_b8,r13 <= 536870910
 0 <= aux_Omega_b8,r10 <= 536870910
 0 <= sv_function_helpervar_b2,r13__Z3 <= 1
 0 <= sv_function_helpervar_b3,r7__Z3 <= 1
 0 <= aux_Omega_b8,r11 <= 536870910
 0 <= aux_cap_helper__b4,r1,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b4,r1,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b4,r1,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_b0,r6,band_LOW_PAIRED <= 536870910
 0 <= X__b5,r12,band_LOW_PAIRED <= 2
 0 <= aux_cap_helper__b9,r11,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b9,r11,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b9,r11,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_b1,r13,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_b0,r6,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b3,r1,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b1,r4,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b1,r4,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b3,r1,band_UNPAIRED__condX1 <= 2
 0 <= X__b2,r10,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b7,r12,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b7,r12,band_LOW_PAIRED__Z1 <= 1
 0 <= X__b0,r5,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b4,r3,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b4,r3,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b2,r8,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b4,r3,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_b6,r3,band_UNPAIRED <= 536870910
 0 <= aux_cap_b7,r12,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_helper__b7,r6,band_UNPAIRED__condX2 <= 2
 0 <= X__b0,r10,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b7,r6,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b2,r8,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b9,r13,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b9,r13,band_HIGH_PAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b5,r6__condX1 <= 536870910
 0 <= aux_cap_helper__b9,r13,band_HIGH_PAIRED__Z2 <= 1
 0 <= X__b4,r8,band_UNPAIRED <= 2
 0 <= sv_function_helpervar_b5,r6__condX2 <= 536870910
 0 <= sv_function_helpervar_b5,r6__condX3 <= 536870910
 0 <= aux_cap_b0,r5,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b7,r3,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b7,r3,band_LOW_PAIRED__Z2 <= 1
 0 <= X__b3,r3,band_LOW_PAIRED <= 2
 0 <= aux_cap_helper__b0,r2,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b1,r10,band_UNPAIRED__condX2 <= 2
 0 <= X__b9,r6,band_LOW_PAIRED <= 2
 0 <= aux_cap_b9,r5,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_helper__b0,r2,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b1,r10,band_UNPAIRED__condX1 <= 2
 0 <= X__b3,r8,band_HIGH_PAIRED <= 3
 0 <= aux_cap_helper__b5,r13,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b8,r8,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b5,r13,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b8,r8,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_b9,r10,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_helper__b0,r8,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b0,r8,band_LOW_PAIRED__Z2 <= 1
 0 <= sv_function_helpervar_b9,r1__condX1 <= 536870910
 0 <= sv_function_helpervar_b9,r1__condX2 <= 536870910
 0 <= aux_cap_helper__b2,r5,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b2,r5,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_b1,r0,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b2,r5,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_b6,r8,band_HIGH_PAIRED <= 536870910
 0 <= sv_function_helpervar_b0,r8__condX1 <= 536870910
 0 <= sv_function_helpervar_b0,r8__condX2 <= 536870910
 0 <= sv_function_helpervar_b0,r8__condX3 <= 536870910
 0 <= X__b9,r7,band_HIGH_PAIRED <= 3
 0 <= aux_cap_b2,r0,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b3,r7,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_b8,r5,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_helper__b3,r3,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b3,r3,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b3,r3,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b3,r7,band_LOW_PAIRED__condX1 <= 2
 0 <= sv_function_helpervar_b2,r12__condX2 <= 536870910
 0 <= X__b6,r10,band_UNPAIRED <= 2
 0 <= sv_function_helpervar_b2,r12__condX3 <= 536870910
 0 <= X__b7,r10,band_LOW_PAIRED <= 2
 0 <= sv_function_helpervar_b2,r12__condX1 <= 536870910
 0 <= aux_cap_b5,r8,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b6,r3,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b6,r3,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_b2,r2,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_helper__b8,r0,band_LOW_PAIRED__Z2 <= 1
 0 <= X__b0,r0,band_HIGH_PAIRED <= 3
 0 <= aux_cap_helper__b4,r0,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b8,r0,band_LOW_PAIRED__Z1 <= 1
 0 <= GlobalBidder_PSI__i7,k2 <= 536870910
 0 <= GlobalBidder_PSI__i7,k1 <= 536870910
 0 <= GlobalBidder_PSI__i7,k0 <= 536870910
 0 <= sv_function_helpervar_b8,r8__Z3 <= 1
 0 <= X__b4,r8,band_HIGH_PAIRED <= 3
 0 <= sv_function_helpervar_b8,r8__Z2 <= 1
 0 <= sv_function_helpervar_b8,r8__Z1 <= 1
 0 <= aux_cap_helper__b1,r2,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b9,r11,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b1,r2,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b9,r11,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b0,r4,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b0,r4,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b0,r4,band_HIGH_PAIRED__Z1 <= 1
 0 <= X__b6,r8,band_LOW_PAIRED <= 2
 0 <= aux_cap_b3,r1,band_UNPAIRED <= 536870910
 0 <= aux_cap_b8,r12,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_b3,r2,band_LOW_PAIRED <= 536870910
 0 <= X__b6,r12,band_LOW_PAIRED <= 2
 0 <= aux_cap_helper__b2,r0,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b2,r0,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_sv_b3,r7 <= 536870910
 0 <= aux_cap_helper__b1,r3,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_sv_b3,r6 <= 536870910
 0 <= aux_cap_helper__b1,r3,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_sv_b3,r5 <= 536870910
 0 <= aux_cap_helper__b4,r13,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_sv_b3,r4 <= 536870910
 0 <= aux_sv_b3,r3 <= 536870910
 0 <= aux_cap_helper__b4,r13,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_sv_b3,r2 <= 536870910
 0 <= aux_cap_helper__b4,r13,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b0,r12,band_UNPAIRED__condX1 <= 2
 0 <= aux_sv_b3,r1 <= 536870910
 0 <= aux_cap_helper__b0,r12,band_UNPAIRED__condX2 <= 2
 0 <= aux_sv_b3,r0 <= 536870910
 0 <= aux_cap_b8,r12,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_b6,r4,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b1,r0,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b1,r0,band_UNPAIRED__Z1 <= 1
 0 <= aux_sv_b3,r9 <= 536870910
 0 <= aux_sv_b3,r8 <= 536870910
 0 <= aux_cap_b5,r7,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_b5,r13,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b0,r10,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b6,r1,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b0,r10,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b2,r5,band_HIGH_PAIRED__Z3 <= 1
 0 <= sv_function_helpervar_b1,r6__Z1 <= 1
 0 <= aux_cap_helper__b6,r1,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b6,r1,band_HIGH_PAIRED__condX2 <= 3
 0 <= sv_function_helpervar_b1,r6__Z3 <= 1
 0 <= sv_function_helpervar_b1,r6__Z2 <= 1
 0 <= aux_cap_helper__b2,r5,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b2,r5,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b0,r10,band_HIGH_PAIRED__Z3 <= 1
 0 <= sv_function_helpervar_b9,r1__condX3 <= 536870910
 0 <= X__b0,r10,band_HIGH_PAIRED <= 3
 0 <= sv_function_helpervar_b0,r7__Z1 <= 1
 0 <= aux_cap_helper__b1,r5,band_LOW_PAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b0,r7__Z2 <= 1
 0 <= aux_cap_helper__b3,r11,band_HIGH_PAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b0,r7__Z3 <= 1
 0 <= aux_cap_helper__b3,r11,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b3,r11,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b1,r5,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b3,r4,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b3,r4,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_b4,r7,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b1,r1,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b1,r1,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b6,r12,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b1,r1,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b3,r8,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b3,r8,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b2,r5,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b2,r5,band_LOW_PAIRED__condX2 <= 2
 0 <= X__b1,r4,band_HIGH_PAIRED <= 3
 0 <= aux_cap_helper__b6,r12,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b6,r12,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b5,r0,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b5,r0,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_b5,r4,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b7,r7,band_LOW_PAIRED__condX2 <= 2
 0 <= X__b6,r3,band_HIGH_PAIRED <= 3
 0 <= X__b0,r5,band_LOW_PAIRED <= 2
 0 <= sv_function_helpervar_b2,r1__Z2 <= 1
 0 <= sv_function_helpervar_b2,r1__Z3 <= 1
 0 <= aux_cap_helper__b7,r7,band_LOW_PAIRED__condX1 <= 2
 0 <= sv_function_helpervar_b2,r1__Z1 <= 1
 0 <= sv_function_helpervar_b7,r8__condX2 <= 536870910
 0 <= sv_function_helpervar_b7,r8__condX3 <= 536870910
 0 <= sv_function_helpervar_b7,r8__condX1 <= 536870910
 0 <= aux_cap_helper__b4,r3,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b4,r3,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_b6,r9,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_b4,r10,band_UNPAIRED <= 536870910
 0 <= aux_cap_b7,r9,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b1,r5,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b1,r5,band_UNPAIRED__condX1 <= 2
 0 <= X__b8,r11,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b2,r1,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b2,r1,band_HIGH_PAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b9,r3__Z3 <= 1
 0 <= aux_cap_helper__b4,r5,band_LOW_PAIRED__Z2 <= 1
 0 <= sv_function_helpervar_b9,r3__Z1 <= 1
 0 <= aux_cap_helper__b2,r1,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b4,r5,band_LOW_PAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b9,r3__Z2 <= 1
 0 <= X__b3,r4,band_HIGH_PAIRED <= 3
 0 <= aux_cap_helper__b9,r8,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b9,r8,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b9,r8,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b2,r7,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b2,r7,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b3,r2,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b6,r8,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b3,r2,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_b3,r4,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b6,r8,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_b8,r3,band_HIGH_PAIRED <= 536870910
 0 <= sv_function_helpervar_b6,r7__Z3 <= 1
 0 <= sv_function_helpervar_b6,r7__Z2 <= 1
 0 <= sv_function_helpervar_b6,r7__Z1 <= 1
 0 <= X__b8,r10,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b7,r6,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b7,r6,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b8,r8,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b8,r8,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b8,r12,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b8,r12,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b8,r12,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_b3,r0,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b5,r2,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b5,r2,band_UNPAIRED__Z1 <= 1
 0 <= X__b7,r12,band_LOW_PAIRED <= 2
 0 <= aux_cap_helper__b6,r1,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b6,r1,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b9,r5,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b6,r8,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b9,r5,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b9,r5,band_HIGH_PAIRED__condX2 <= 3
 0 <= sv_function_helpervar_b0,r7__condX1 <= 536870910
 0 <= sv_function_helpervar_b0,r7__condX2 <= 536870910
 0 <= aux_cap_b3,r13,band_HIGH_PAIRED <= 536870910
 0 <= sv_function_helpervar_b0,r7__condX3 <= 536870910
 0 <= aux_cap_helper__b3,r9,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_Omega_b4,r12 <= 536870910
 0 <= aux_Omega_b4,r13 <= 536870910
 0 <= aux_Omega_b4,r10 <= 536870910
 0 <= aux_cap_helper__b6,r8,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b3,r9,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_Omega_b4,r11 <= 536870910
 0 <= aux_cap_b7,r5,band_LOW_PAIRED <= 536870910
 0 <= X__b7,r2,band_UNPAIRED <= 2
 0 <= aux_cap_b9,r12,band_LOW_PAIRED <= 536870910
 0 <= X__b8,r13,band_HIGH_PAIRED <= 3
 0 <= aux_cap_b5,r9,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_b8,r3,band_LOW_PAIRED <= 536870910
 0 <= X__b9,r4,band_LOW_PAIRED <= 2
 0 <= aux_cap_helper__b7,r8,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b7,r8,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b0,r10,band_UNPAIRED__Z2 <= 1
 0 <= X__b8,r8,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b0,r10,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_b4,r13,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b9,r8,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b9,r8,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_b2,r2,band_UNPAIRED <= 536870910
 0 <= X__b8,r6,band_LOW_PAIRED <= 2
 0 <= aux_cap_b5,r13,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_b8,r0,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b4,r2,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b4,r2,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b9,r10,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b9,r10,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b9,r10,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b5,r6,band_UNPAIRED__Z1 <= 1
 0 <= aux_sv_b6,r3 <= 536870910
 0 <= aux_cap_helper__b5,r6,band_UNPAIRED__Z2 <= 1
 0 <= aux_sv_b6,r4 <= 536870910
 0 <= aux_sv_b6,r5 <= 536870910
 0 <= aux_sv_b6,r6 <= 536870910
 0 <= aux_sv_b6,r7 <= 536870910
 0 <= aux_sv_b6,r8 <= 536870910
 0 <= aux_sv_b6,r9 <= 536870910
 0 <= aux_cap_b8,r10,band_UNPAIRED <= 536870910
 0 <= aux_sv_b6,r0 <= 536870910
 0 <= aux_sv_b6,r1 <= 536870910
 0 <= aux_sv_b6,r2 <= 536870910
 0 <= aux_cap_helper__b5,r8,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_b6,r13,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b5,r8,band_UNPAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b8,r5__condX1 <= 536870910
 0 <= sv_function_helpervar_b8,r5__condX2 <= 536870910
 0 <= sv_function_helpervar_b8,r5__condX3 <= 536870910
 0 <= aux_cap_b6,r10,band_UNPAIRED <= 536870910
 0 <= sv_function_helpervar_b3,r3__Z2 <= 1
 0 <= X__b7,r7,band_HIGH_PAIRED <= 3
 0 <= sv_function_helpervar_b3,r3__Z1 <= 1
 0 <= aux_cap_helper__b4,r6,band_UNPAIRED__Z2 <= 1
 0 <= sv_function_helpervar_b3,r3__Z3 <= 1
 0 <= aux_cap_helper__b4,r6,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_b6,r7,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_b0,r5,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_b7,r2,band_UNPAIRED <= 536870910
 0 <= X__b8,r7,band_HIGH_PAIRED <= 3
 0 <= aux_cap_helper__b5,r4,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b9,r13,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b5,r4,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b9,r13,band_HIGH_PAIRED__condX2 <= 3
 0 <= X__b1,r4,band_UNPAIRED <= 2
 0 <= aux_cap_b7,r8,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b6,r0,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b6,r0,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_b1,r2,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_b1,r11,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_helper__b6,r0,band_HIGH_PAIRED__Z1 <= 1
 0 <= X__b3,r9,band_UNPAIRED <= 2
 0 <= aux_cap_b3,r13,band_UNPAIRED <= 536870910
 0 <= X__b5,r10,band_LOW_PAIRED <= 2
 0 <= GlobalBidder_W_ir__b7,r13 <= 1
 0 <= X__b3,r1,band_LOW_PAIRED <= 2
 0 <= GlobalBidder_W_ir__b7,r11 <= 1
 0 <= GlobalBidder_W_ir__b7,r12 <= 1
 0 <= aux_cap_helper__b2,r11,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b2,r11,band_HIGH_PAIRED__condX2 <= 3
 0 <= GlobalBidder_W_ir__b7,r10 <= 1
 0 <= aux_cap_helper__b5,r3,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b1,r4,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b2,r11,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b1,r4,band_UNPAIRED__condX2 <= 2
 0 <= sv_function_helpervar_b5,r8__Z1 <= 1
 0 <= sv_function_helpervar_b5,r8__Z2 <= 1
 0 <= aux_cap_helper__b2,r4,band_HIGH_PAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b5,r8__Z3 <= 1
 0 <= aux_cap_helper__b6,r2,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b5,r3,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b6,r10,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_b8,r10,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_helper__b5,r3,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b6,r2,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b6,r10,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b9,r13,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b2,r4,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b5,r12,band_LOW_PAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b5,r7__condX1 <= 536870910
 0 <= aux_cap_helper__b2,r4,band_HIGH_PAIRED__Z3 <= 1
 0 <= sv_function_helpervar_b5,r7__condX2 <= 536870910
 0 <= sv_function_helpervar_b5,r7__condX3 <= 536870910
 0 <= aux_cap_helper__b5,r12,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b0,r13,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b4,r2,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b4,r2,band_UNPAIRED__Z1 <= 1
 0 <= X__b8,r2,band_HIGH_PAIRED <= 3
 0 <= aux_cap_helper__b4,r5,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b4,r5,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b6,r6,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b6,r6,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b9,r8,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b9,r8,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b0,r13,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_b9,r7,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_b9,r6,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b4,r10,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b4,r10,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_b3,r8,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b5,r11,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b5,r11,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b9,r8,band_HIGH_PAIRED__condX3 <= 3
 0 <= X__b0,r7,band_LOW_PAIRED <= 2
 0 <= aux_cap_helper__b0,r6,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b0,r6,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b0,r6,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_b7,r10,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_helper__b2,r2,band_HIGH_PAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b7,r11__Z1 <= 1
 0 <= aux_cap_helper__b0,r4,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b2,r2,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b2,r2,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b0,r9,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b3,r13,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b0,r9,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b3,r13,band_LOW_PAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b7,r11__Z3 <= 1
 0 <= sv_function_helpervar_b7,r11__Z2 <= 1
 0 <= X__b5,r8,band_HIGH_PAIRED <= 3
 0 <= aux_cap_helper__b0,r4,band_LOW_PAIRED__condX2 <= 2
 0 <= X__b6,r10,band_LOW_PAIRED <= 2
 0 <= aux_cap_helper__b9,r2,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b9,r2,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b9,r2,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b7,r5,band_UNPAIRED__condX1 <= 2
 0 <= GlobalBidder_hat{W}__i7,k0 <= 1
 0 <= GlobalBidder_hat{W}__i7,k1 <= 1
 0 <= GlobalBidder_hat{W}__i7,k2 <= 1
 0 <= aux_cap_helper__b6,r7,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b6,r7,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b6,r7,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b7,r5,band_UNPAIRED__condX2 <= 2
 0 <= sv_function_helpervar_b1,r2__Z1 <= 1
 0 <= sv_function_helpervar_b1,r2__Z3 <= 1
 0 <= aux_cap_helper__b3,r0,band_UNPAIRED__Z2 <= 1
 0 <= X__b0,r13,band_LOW_PAIRED <= 2
 0 <= sv_function_helpervar_b1,r2__Z2 <= 1
 0 <= aux_cap_helper__b3,r0,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b2,r0,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b2,r0,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b3,r2,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b3,r2,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b9,r9,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b5,r0,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b5,r0,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b7,r9,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b7,r9,band_LOW_PAIRED__condX1 <= 2
 0 <= X__b9,r7,band_UNPAIRED <= 2
 0 <= aux_sv_b0,r10 <= 536870910
 0 <= aux_sv_b0,r11 <= 536870910
 0 <= aux_sv_b0,r12 <= 536870910
 0 <= aux_sv_b0,r13 <= 536870910
 0 <= aux_cap_helper__b5,r0,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_b9,r3,band_LOW_PAIRED <= 536870910
 0 <= sv_function_helpervar_b6,r3__condX2 <= 536870910
 0 <= sv_function_helpervar_b6,r3__condX3 <= 536870910
 0 <= sv_function_helpervar_b6,r3__condX1 <= 536870910
 0 <= sv_function_helpervar_b8,r4__Z3 <= 1
 0 <= sv_function_helpervar_b8,r4__Z2 <= 1
 0 <= aux_c_b1,r13 <= 536870910
 0 <= sv_function_helpervar_b8,r4__Z1 <= 1
 0 <= aux_c_b1,r12 <= 536870910
 0 <= aux_c_b1,r11 <= 536870910
 0 <= X__b6,r4,band_UNPAIRED <= 2
 0 <= aux_c_b1,r10 <= 536870910
 0 <= X__b1,r9,band_HIGH_PAIRED <= 3
 0 <= sv_function_helpervar_b4,r10__Z2 <= 1
 0 <= aux_cap_helper__b9,r9,band_HIGH_PAIRED__Z3 <= 1
 0 <= sv_function_helpervar_b4,r10__Z3 <= 1
 0 <= aux_cap_helper__b9,r9,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b9,r9,band_HIGH_PAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b6,r3__Z1 <= 1
 0 <= sv_function_helpervar_b4,r10__condX1 <= 536870910
 0 <= sv_function_helpervar_b7,r10__condX3 <= 536870910
 0 <= X__b4,r7,band_UNPAIRED <= 2
 0 <= sv_function_helpervar_b1,r3__condX1 <= 536870910
 0 <= sv_function_helpervar_b4,r10__condX3 <= 536870910
 0 <= sv_function_helpervar_b1,r3__condX2 <= 536870910
 0 <= sv_function_helpervar_b4,r10__condX2 <= 536870910
 0 <= sv_function_helpervar_b1,r3__condX3 <= 536870910
 0 <= aux_cap_b2,r13,band_LOW_PAIRED <= 536870910
 0 <= sv_function_helpervar_b7,r10__condX1 <= 536870910
 0 <= sv_function_helpervar_b7,r10__condX2 <= 536870910
 0 <= aux_cap_helper__b3,r9,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b3,r9,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b9,r8,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b9,r8,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b7,r4,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b7,r4,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_b3,r9,band_HIGH_PAIRED <= 536870910
 0 <= X__b2,r3,band_LOW_PAIRED <= 2
 0 <= aux_cap_helper__b3,r12,band_HIGH_PAIRED__condX3 <= 3
 0 <= aux_cap_helper__b3,r12,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b3,r12,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b7,r11,band_LOW_PAIRED__Z2 <= 1
 0 <= sv_function_helpervar_b4,r10__Z1 <= 1
 0 <= aux_cap_helper__b7,r11,band_LOW_PAIRED__Z1 <= 1
 0 <= X__b0,r6,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b1,r12,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b1,r12,band_UNPAIRED__Z2 <= 1
 0 <= sv_function_helpervar_b3,r5__condX1 <= 536870910
 0 <= sv_function_helpervar_b3,r5__condX2 <= 536870910
 0 <= aux_cap_b0,r4,band_LOW_PAIRED <= 536870910
 0 <= sv_function_helpervar_b3,r5__condX3 <= 536870910
 0 <= X__b4,r10,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b2,r3,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b2,r3,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_cap_helper__b2,r3,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_b8,r8,band_UNPAIRED <= 536870910
 0 <= X__b5,r3,band_HIGH_PAIRED <= 3
 0 <= aux_cap_helper__b2,r3,band_LOW_PAIRED__condX1 <= 2
 0 <= sv_function_helpervar_b2,r13__condX1 <= 536870910
 0 <= aux_cap_helper__b2,r3,band_LOW_PAIRED__condX2 <= 2
 0 <= sv_function_helpervar_b2,r13__condX2 <= 536870910
 0 <= sv_function_helpervar_b2,r13__condX3 <= 536870910
 0 <= aux_cap_b3,r0,band_LOW_PAIRED <= 536870910
 0 <= aux_cap_helper__b4,r4,band_UNPAIRED__Z1 <= 1
 0 <= aux_cap_helper__b4,r4,band_UNPAIRED__Z2 <= 1
 0 <= aux_cap_helper__b6,r10,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b6,r10,band_HIGH_PAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b6,r3__Z3 <= 1
 0 <= aux_cap_helper__b9,r2,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b6,r10,band_HIGH_PAIRED__Z2 <= 1
 0 <= sv_function_helpervar_b6,r3__Z2 <= 1
 0 <= aux_cap_helper__b9,r2,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_b4,r4,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b6,r9,band_LOW_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b6,r9,band_LOW_PAIRED__Z1 <= 1
 0 <= aux_cap_b1,r13,band_UNPAIRED <= 536870910
 0 <= aux_Omega_b8,r5 <= 536870910
 0 <= aux_Omega_b8,r6 <= 536870910
 0 <= aux_Omega_b8,r7 <= 536870910
 0 <= X__b1,r5,band_LOW_PAIRED <= 2
 0 <= aux_Omega_b8,r8 <= 536870910
 0 <= aux_Omega_b8,r9 <= 536870910
 0 <= sv_function_helpervar_b2,r9__condX1 <= 536870910
 0 <= sv_function_helpervar_b2,r9__condX2 <= 536870910
 0 <= sv_function_helpervar_b2,r9__condX3 <= 536870910
 0 <= aux_cap_b7,r3,band_HIGH_PAIRED <= 536870910
 0 <= aux_cap_helper__b4,r4,band_LOW_PAIRED__Z1 <= 1
 0 <= sv_function_helpervar_b4,r1__condX1 <= 536870910
 0 <= sv_function_helpervar_b4,r1__condX2 <= 536870910
 0 <= sv_function_helpervar_b4,r1__condX3 <= 536870910
 0 <= X__b9,r13,band_HIGH_PAIRED <= 3
 0 <= X__b7,r8,band_LOW_PAIRED <= 2
 0 <= aux_c_b9,r1 <= 536870910
 0 <= aux_c_b9,r0 <= 536870910
 0 <= aux_c_b9,r3 <= 536870910
 0 <= aux_c_b9,r2 <= 536870910
 0 <= aux_c_b9,r5 <= 536870910
 0 <= aux_c_b9,r4 <= 536870910
 0 <= aux_c_b9,r7 <= 536870910
 0 <= aux_c_b9,r6 <= 536870910
 0 <= aux_c_b9,r9 <= 536870910
 0 <= aux_cap_helper__b0,r13,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_c_b9,r8 <= 536870910
 0 <= aux_cap_helper__b0,r13,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_b5,r5,band_UNPAIRED <= 536870910
 0 <= aux_cap_helper__b8,r2,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b8,r2,band_LOW_PAIRED__condX1 <= 2
 0 <= aux_cap_helper__b4,r4,band_LOW_PAIRED__Z2 <= 1
 0 <= X__b2,r4,band_HIGH_PAIRED <= 3
 0 <= aux_Omega_b8,r0 <= 536870910
 0 <= X__b3,r1,band_UNPAIRED <= 2
 0 <= aux_cap_helper__b0,r5,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_Omega_b8,r1 <= 536870910
 0 <= aux_cap_helper__b0,r5,band_HIGH_PAIRED__Z1 <= 1
 0 <= aux_Omega_b8,r2 <= 536870910
 0 <= aux_Omega_b8,r3 <= 536870910
 0 <= aux_cap_helper__b0,r5,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_Omega_b8,r4 <= 536870910
Binaries
 aux_cap_helper__b5,r5,band_LOW_PAIRED__Z1 
 aux_cap_helper__b2,r10,band_UNPAIRED__Z1 
 aux_cap_helper__b4,r10,band_LOW_PAIRED__Z2 
 aux_cap_helper__b2,r10,band_UNPAIRED__Z2 
 aux_cap_helper__b4,r10,band_LOW_PAIRED__Z1 
 aux_cap_helper__b2,r12,band_LOW_PAIRED__Z1 
 aux_cap_helper__b2,r12,band_LOW_PAIRED__Z2 
 aux_cap_helper__b3,r11,band_LOW_PAIRED__Z2 
 aux_cap_helper__b3,r11,band_LOW_PAIRED__Z1  sv_function_helpervar_b0,r10__Z2 
 aux_cap_helper__b1,r13,band_LOW_PAIRED__Z2  sv_function_helpervar_b0,r10__Z3 
 aux_cap_helper__b1,r13,band_LOW_PAIRED__Z1  sv_function_helpervar_b0,r10__Z1 
 aux_cap_helper__b4,r4,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b4,r4,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b4,r4,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b2,r12,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b2,r12,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b2,r12,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b3,r2,band_LOW_PAIRED__Z1 
 aux_cap_helper__b3,r2,band_LOW_PAIRED__Z2 
 aux_cap_helper__b9,r5,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b9,r5,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b9,r5,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b3,r6,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b3,r6,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b3,r6,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b7,r9,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b7,r9,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b7,r9,band_HIGH_PAIRED__Z3  sv_function_helpervar_b5,r3__Z2 
 sv_function_helpervar_b5,r3__Z1  sv_function_helpervar_b5,r3__Z3 
 aux_cap_helper__b5,r5,band_LOW_PAIRED__Z2  sv_function_helpervar_b9,r12__Z3 
 sv_function_helpervar_b9,r12__Z1  sv_function_helpervar_b9,r12__Z2 
 aux_cap_helper__b7,r8,band_LOW_PAIRED__Z1 
 aux_cap_helper__b7,r8,band_LOW_PAIRED__Z2 
 aux_cap_helper__b4,r7,band_LOW_PAIRED__Z1 
 aux_cap_helper__b4,r7,band_LOW_PAIRED__Z2 
 aux_cap_helper__b2,r8,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b9,r0,band_LOW_PAIRED__Z1 
 aux_cap_helper__b9,r0,band_LOW_PAIRED__Z2 
 aux_cap_helper__b2,r8,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b2,r8,band_HIGH_PAIRED__Z2  sv_function_helpervar_b3,r2__Z1 
 sv_function_helpervar_b3,r2__Z3  sv_function_helpervar_b3,r2__Z2 
 aux_cap_helper__b8,r10,band_HIGH_PAIRED__Z3  sv_function_helpervar_b0,r4__Z1 
 sv_function_helpervar_b0,r4__Z2  sv_function_helpervar_b0,r4__Z3 
 aux_cap_helper__b5,r2,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b5,r2,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b5,r2,band_HIGH_PAIRED__Z3  sv_function_helpervar_b1,r8__Z3 
 sv_function_helpervar_b1,r8__Z2  sv_function_helpervar_b1,r8__Z1 
 aux_cap_helper__b2,r4,band_LOW_PAIRED__Z2 
 aux_cap_helper__b7,r11,band_UNPAIRED__Z1 
 aux_cap_helper__b7,r11,band_UNPAIRED__Z2 
 aux_cap_helper__b7,r11,band_HIGH_PAIRED__Z2  sv_function_helpervar_b7,r8__Z2 
 aux_cap_helper__b2,r4,band_LOW_PAIRED__Z1 
 aux_cap_helper__b7,r11,band_HIGH_PAIRED__Z1  sv_function_helpervar_b7,r8__Z3 
 sv_function_helpervar_b7,r8__Z1  sv_function_helpervar_b6,r10__Z1 
 sv_function_helpervar_b6,r10__Z3  sv_function_helpervar_b6,r10__Z2 
 aux_cap_helper__b8,r10,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b8,r10,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b6,r4,band_UNPAIRED__Z2 
 aux_cap_helper__b6,r4,band_UNPAIRED__Z1  sv_function_helpervar_b8,r1__Z3 
 sv_function_helpervar_b8,r1__Z2  sv_function_helpervar_b8,r1__Z1 
 sv_function_helpervar_b2,r9__Z2  sv_function_helpervar_b2,r9__Z3 
 sv_function_helpervar_b2,r9__Z1  aux_cap_helper__b1,r1,band_UNPAIRED__Z1 
 aux_cap_helper__b1,r1,band_UNPAIRED__Z2 
 aux_cap_helper__b8,r13,band_LOW_PAIRED__Z2 
 aux_cap_helper__b8,r13,band_LOW_PAIRED__Z1 
 aux_cap_helper__b4,r9,band_UNPAIRED__Z1 
 aux_cap_helper__b4,r9,band_UNPAIRED__Z2 
 aux_cap_helper__b8,r5,band_UNPAIRED__Z2 
 aux_cap_helper__b8,r5,band_UNPAIRED__Z1 
 aux_cap_helper__b7,r1,band_LOW_PAIRED__Z2 
 aux_cap_helper__b7,r1,band_LOW_PAIRED__Z1 
 aux_cap_helper__b6,r8,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b6,r8,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b6,r8,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b2,r8,band_LOW_PAIRED__Z2 
 aux_cap_helper__b2,r0,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b2,r0,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b2,r8,band_LOW_PAIRED__Z1 
 aux_cap_helper__b2,r0,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b8,r2,band_UNPAIRED__Z1 
 aux_cap_helper__b9,r4,band_UNPAIRED__Z2 
 aux_cap_helper__b9,r4,band_UNPAIRED__Z1 
 aux_cap_helper__b8,r2,band_UNPAIRED__Z2 
 aux_cap_helper__b1,r7,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b0,r13,band_UNPAIRED__Z1 
 aux_cap_helper__b1,r7,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b0,r13,band_UNPAIRED__Z2 
 aux_cap_helper__b1,r7,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b9,r9,band_UNPAIRED__Z1 
 aux_cap_helper__b9,r9,band_UNPAIRED__Z2 
 aux_cap_helper__b0,r9,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b9,r11,band_UNPAIRED__Z2 
 aux_cap_helper__b0,r9,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b2,r8,band_UNPAIRED__Z1 
 aux_cap_helper__b9,r11,band_UNPAIRED__Z1 
 aux_cap_helper__b0,r9,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b2,r8,band_UNPAIRED__Z2 
 aux_cap_helper__b3,r7,band_UNPAIRED__Z1 
 aux_cap_helper__b3,r7,band_UNPAIRED__Z2 
 aux_cap_helper__b6,r3,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b1,r13,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b6,r3,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b1,r13,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b6,r3,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b1,r13,band_HIGH_PAIRED__Z3  sv_function_helpervar_b4,r6__Z3 
 sv_function_helpervar_b4,r6__Z1  sv_function_helpervar_b4,r6__Z2 
 sv_function_helpervar_b9,r9__Z3  sv_function_helpervar_b9,r9__Z2 
 sv_function_helpervar_b9,r9__Z1  aux_cap_helper__b6,r3,band_LOW_PAIRED__Z1 
 aux_cap_helper__b6,r3,band_LOW_PAIRED__Z2 
 aux_cap_helper__b1,r4,band_UNPAIRED__Z2 
 aux_cap_helper__b1,r4,band_UNPAIRED__Z1 
 aux_cap_helper__b0,r12,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b0,r12,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b0,r12,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b0,r2,band_UNPAIRED__Z2 
 aux_cap_helper__b5,r0,band_UNPAIRED__Z1 
 aux_cap_helper__b5,r0,band_UNPAIRED__Z2 
 aux_cap_helper__b0,r2,band_UNPAIRED__Z1 
 aux_cap_helper__b9,r7,band_LOW_PAIRED__Z1 
 aux_cap_helper__b9,r7,band_LOW_PAIRED__Z2 
 aux_cap_helper__b5,r11,band_HIGH_PAIRED__Z3  sv_function_helpervar_b1,r13__Z1 
 aux_cap_helper__b5,r11,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b5,r11,band_HIGH_PAIRED__Z1  sv_function_helpervar_b3,r12__Z1 
 sv_function_helpervar_b3,r12__Z2  sv_function_helpervar_b3,r12__Z3 
 sv_function_helpervar_b1,r13__Z3  sv_function_helpervar_b1,r13__Z2 
 aux_cap_helper__b0,r1,band_HIGH_PAIRED__Z3  GlobalBidder_hat{W}__i8,k0 
 GlobalBidder_hat{W}__i8,k1  aux_cap_helper__b0,r1,band_HIGH_PAIRED__Z1 
 GlobalBidder_hat{W}__i8,k2  aux_cap_helper__b0,r1,band_HIGH_PAIRED__Z2 
 sv_function_helpervar_b0,r0__Z3  sv_function_helpervar_b0,r0__Z2 
 sv_function_helpervar_b0,r0__Z1  aux_cap_helper__b1,r6,band_LOW_PAIRED__Z2 
 aux_cap_helper__b1,r6,band_LOW_PAIRED__Z1 
 aux_cap_helper__b7,r3,band_UNPAIRED__Z2 
 aux_cap_helper__b7,r3,band_UNPAIRED__Z1 
 aux_cap_helper__b7,r1,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b7,r1,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b7,r1,band_HIGH_PAIRED__Z1  sv_function_helpervar_b2,r5__Z2 
 sv_function_helpervar_b2,r5__Z3  sv_function_helpervar_b7,r4__Z1 
 sv_function_helpervar_b2,r5__Z1  aux_cap_helper__b8,r7,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b8,r7,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b8,r7,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b2,r1,band_LOW_PAIRED__Z1 
 aux_cap_helper__b2,r1,band_LOW_PAIRED__Z2 
 aux_cap_helper__b9,r12,band_LOW_PAIRED__Z2 
 aux_cap_helper__b9,r12,band_LOW_PAIRED__Z1 
 aux_cap_helper__b4,r10,band_UNPAIRED__Z2 
 aux_cap_helper__b4,r10,band_UNPAIRED__Z1  sv_function_helpervar_b7,r4__Z2 
 sv_function_helpervar_b7,r4__Z3  sv_function_helpervar_b8,r9__Z3 
 sv_function_helpervar_b8,r9__Z2  sv_function_helpervar_b8,r9__Z1 
 aux_cap_helper__b3,r9,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b3,r9,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b3,r9,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b9,r2,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b9,r2,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b9,r2,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b5,r9,band_LOW_PAIRED__Z2 
 aux_cap_helper__b5,r9,band_LOW_PAIRED__Z1 
 aux_cap_helper__b5,r1,band_LOW_PAIRED__Z2 
 aux_cap_helper__b5,r1,band_LOW_PAIRED__Z1 
 aux_cap_helper__b4,r7,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b4,r7,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b4,r7,band_HIGH_PAIRED__Z3  sv_function_helpervar_b7,r10__Z3 
 sv_function_helpervar_b7,r10__Z2  sv_function_helpervar_b7,r10__Z1 
 sv_function_helpervar_b7,r1__Z1  sv_function_helpervar_b7,r1__Z2 
 sv_function_helpervar_b7,r1__Z3  sv_function_helpervar_b1,r5__Z2 
 sv_function_helpervar_b1,r5__Z1  sv_function_helpervar_b1,r5__Z3 
 aux_cap_helper__b0,r10,band_LOW_PAIRED__Z1 
 aux_cap_helper__b0,r10,band_LOW_PAIRED__Z2  sv_function_helpervar_b0,r8__Z3 
 sv_function_helpervar_b0,r8__Z1  sv_function_helpervar_b0,r8__Z2 
 aux_cap_helper__b9,r0,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b9,r0,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b9,r0,band_HIGH_PAIRED__Z1  sv_function_helpervar_b1,r10__Z3 
 sv_function_helpervar_b1,r10__Z2  sv_function_helpervar_b1,r10__Z1 
 aux_cap_helper__b3,r6,band_LOW_PAIRED__Z1 
 aux_cap_helper__b3,r6,band_LOW_PAIRED__Z2 
 aux_cap_helper__b4,r9,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b4,r9,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b4,r9,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b0,r7,band_UNPAIRED__Z1 
 aux_cap_helper__b5,r2,band_LOW_PAIRED__Z2  sv_function_helpervar_b9,r2__Z2 
 aux_cap_helper__b0,r7,band_UNPAIRED__Z2 
 aux_cap_helper__b5,r2,band_LOW_PAIRED__Z1  sv_function_helpervar_b9,r2__Z3 
 sv_function_helpervar_b9,r2__Z1  aux_cap_helper__b0,r5,band_UNPAIRED__Z2 
 aux_cap_helper__b0,r5,band_UNPAIRED__Z1 
 aux_cap_helper__b7,r0,band_UNPAIRED__Z1 
 aux_cap_helper__b7,r0,band_UNPAIRED__Z2  sv_function_helpervar_b6,r8__Z2 
 sv_function_helpervar_b6,r8__Z1  sv_function_helpervar_b6,r8__Z3 
 aux_cap_helper__b7,r6,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b7,r6,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b7,r6,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b1,r10,band_LOW_PAIRED__Z1 
 aux_cap_helper__b1,r10,band_LOW_PAIRED__Z2  sv_function_helpervar_b5,r11__Z3 
 sv_function_helpervar_b5,r11__Z2  sv_function_helpervar_b5,r11__Z1 
 aux_cap_helper__b9,r4,band_LOW_PAIRED__Z2 
 aux_cap_helper__b9,r4,band_LOW_PAIRED__Z1  sv_function_helpervar_b4,r3__Z1 
 sv_function_helpervar_b4,r3__Z2  sv_function_helpervar_b4,r3__Z3 
 aux_cap_helper__b4,r13,band_UNPAIRED__Z2 
 aux_cap_helper__b8,r10,band_UNPAIRED__Z2 
 aux_cap_helper__b8,r10,band_UNPAIRED__Z1 
 aux_cap_helper__b4,r13,band_UNPAIRED__Z1 
 aux_cap_helper__b8,r2,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b6,r6,band_LOW_PAIRED__Z1 
 aux_cap_helper__b8,r2,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b8,r2,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b5,r5,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b5,r5,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b5,r5,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b6,r6,band_LOW_PAIRED__Z2 
 aux_cap_helper__b8,r9,band_LOW_PAIRED__Z2 
 aux_cap_helper__b8,r9,band_LOW_PAIRED__Z1 
 aux_cap_helper__b4,r12,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b4,r12,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b4,r12,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b6,r6,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b6,r6,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b6,r6,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b6,r5,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b6,r5,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b1,r10,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b1,r10,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b1,r10,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b6,r5,band_HIGH_PAIRED__Z3  sv_function_helpervar_b5,r7__Z3 
 sv_function_helpervar_b5,r7__Z1  sv_function_helpervar_b5,r7__Z2 
 aux_cap_helper__b8,r2,band_LOW_PAIRED__Z2  sv_function_helpervar_b8,r5__Z1 
 aux_cap_helper__b8,r2,band_LOW_PAIRED__Z1 
 aux_cap_helper__b4,r10,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b4,r10,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b4,r10,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b0,r5,band_LOW_PAIRED__Z1  sv_function_helpervar_b8,r5__Z3 
 aux_cap_helper__b0,r5,band_LOW_PAIRED__Z2 
 aux_cap_helper__b7,r11,band_HIGH_PAIRED__Z3  sv_function_helpervar_b8,r5__Z2 
 aux_cap_helper__b1,r11,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b1,r11,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b1,r11,band_HIGH_PAIRED__Z2  sv_function_helpervar_b1,r1__Z3 
 sv_function_helpervar_b1,r1__Z1  sv_function_helpervar_b1,r1__Z2 
 aux_cap_helper__b1,r9,band_UNPAIRED__Z2 
 aux_cap_helper__b1,r9,band_UNPAIRED__Z1 
 aux_cap_helper__b5,r8,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b8,r0,band_UNPAIRED__Z1 
 aux_cap_helper__b8,r0,band_UNPAIRED__Z2 
 aux_cap_helper__b5,r8,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b5,r8,band_HIGH_PAIRED__Z2  sv_function_helpervar_b0,r13__Z2 
 aux_cap_helper__b8,r12,band_UNPAIRED__Z1  sv_function_helpervar_b0,r13__Z1 
 aux_cap_helper__b8,r12,band_UNPAIRED__Z2 
 aux_cap_helper__b0,r8,band_UNPAIRED__Z2  sv_function_helpervar_b0,r13__Z3 
 aux_cap_helper__b0,r8,band_UNPAIRED__Z1 
 aux_cap_helper__b9,r1,band_UNPAIRED__Z1 
 aux_cap_helper__b9,r1,band_UNPAIRED__Z2 
 aux_cap_helper__b7,r4,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b0,r11,band_LOW_PAIRED__Z1 
 aux_cap_helper__b7,r4,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b7,r4,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b3,r5,band_LOW_PAIRED__Z1 
 aux_cap_helper__b0,r11,band_LOW_PAIRED__Z2 
 aux_cap_helper__b3,r5,band_LOW_PAIRED__Z2  sv_function_helpervar_b4,r11__Z3 
 sv_function_helpervar_b4,r11__Z1  sv_function_helpervar_b4,r11__Z2 
 sv_function_helpervar_b6,r0__Z1  sv_function_helpervar_b6,r0__Z2 
 sv_function_helpervar_b6,r0__Z3  sv_function_helpervar_b6,r4__Z2 
 sv_function_helpervar_b6,r4__Z1  sv_function_helpervar_b6,r4__Z3 
 aux_cap_helper__b4,r0,band_LOW_PAIRED__Z1 
 aux_cap_helper__b4,r0,band_LOW_PAIRED__Z2 
 aux_cap_helper__b8,r4,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b8,r4,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b8,r4,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b5,r7,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b5,r7,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b5,r7,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b7,r0,band_LOW_PAIRED__Z1 
 aux_cap_helper__b7,r0,band_LOW_PAIRED__Z2  sv_function_helpervar_b3,r6__Z1 
 sv_function_helpervar_b3,r6__Z3  sv_function_helpervar_b3,r6__Z2 
 sv_function_helpervar_b2,r12__Z3  sv_function_helpervar_b2,r12__Z2 
 sv_function_helpervar_b2,r12__Z1  aux_cap_helper__b3,r11,band_UNPAIRED__Z1 
 aux_cap_helper__b3,r11,band_UNPAIRED__Z2 
 aux_cap_helper__b1,r0,band_LOW_PAIRED__Z2  sv_function_helpervar_b3,r11__Z2 
 aux_cap_helper__b1,r0,band_LOW_PAIRED__Z1  sv_function_helpervar_b3,r11__Z3 
 aux_cap_helper__b8,r9,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b4,r8,band_LOW_PAIRED__Z1 
 aux_cap_helper__b8,r9,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b8,r9,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b4,r8,band_LOW_PAIRED__Z2  sv_function_helpervar_b7,r7__Z1 
 sv_function_helpervar_b7,r7__Z2  sv_function_helpervar_b7,r7__Z3 
 sv_function_helpervar_b3,r11__Z1  sv_function_helpervar_b2,r6__Z1 
 sv_function_helpervar_b2,r6__Z2  sv_function_helpervar_b2,r6__Z3 
 sv_function_helpervar_b8,r2__Z3  sv_function_helpervar_b8,r2__Z2 
 sv_function_helpervar_b8,r2__Z1  aux_cap_helper__b8,r5,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b8,r5,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b8,r5,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b6,r2,band_LOW_PAIRED__Z2 
 aux_cap_helper__b6,r2,band_LOW_PAIRED__Z1 
 aux_cap_helper__b3,r3,band_LOW_PAIRED__Z1 
 aux_cap_helper__b0,r13,band_LOW_PAIRED__Z1 
 aux_cap_helper__b0,r13,band_LOW_PAIRED__Z2 
 aux_cap_helper__b5,r13,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b3,r3,band_LOW_PAIRED__Z2 
 aux_cap_helper__b3,r8,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b3,r8,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b3,r8,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b3,r0,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b3,r0,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b3,r0,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b5,r4,band_LOW_PAIRED__Z2 
 aux_cap_helper__b5,r4,band_LOW_PAIRED__Z1 
 aux_cap_helper__b2,r13,band_LOW_PAIRED__Z2 
 aux_cap_helper__b2,r13,band_LOW_PAIRED__Z1  sv_function_helpervar_b9,r13__Z2 
 sv_function_helpervar_b9,r13__Z3  aux_cap_helper__b2,r11,band_LOW_PAIRED__Z2 
 aux_cap_helper__b2,r11,band_LOW_PAIRED__Z1  sv_function_helpervar_b9,r13__Z1 
 aux_cap_helper__b5,r13,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b8,r3,band_LOW_PAIRED__Z2 
 aux_cap_helper__b5,r13,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b8,r3,band_LOW_PAIRED__Z1  sv_function_helpervar_b5,r6__Z1 
 sv_function_helpervar_b5,r6__Z2  sv_function_helpervar_b5,r6__Z3 
 aux_cap_helper__b1,r7,band_UNPAIRED__Z1 
 aux_cap_helper__b1,r7,band_UNPAIRED__Z2 
 aux_cap_helper__b8,r5,band_LOW_PAIRED__Z2 
 aux_cap_helper__b8,r5,band_LOW_PAIRED__Z1 
 aux_cap_helper__b3,r4,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b3,r4,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b3,r4,band_HIGH_PAIRED__Z3  sv_function_helpervar_b3,r1__Z3 
 aux_cap_helper__b7,r3,band_HIGH_PAIRED__Z1  sv_function_helpervar_b3,r1__Z2 
 aux_cap_helper__b7,r3,band_HIGH_PAIRED__Z2  sv_function_helpervar_b3,r1__Z1 
 aux_cap_helper__b7,r3,band_HIGH_PAIRED__Z3  GlobalBidder_W_ir__b9,r11 
 GlobalBidder_W_ir__b9,r12  aux_cap_helper__b1,r9,band_LOW_PAIRED__Z2 
 GlobalBidder_W_ir__b9,r10  aux_cap_helper__b1,r9,band_LOW_PAIRED__Z1 
 GlobalBidder_W_ir__b9,r13  aux_cap_helper__b3,r5,band_UNPAIRED__Z1 
 aux_cap_helper__b9,r3,band_UNPAIRED__Z1  sv_function_helpervar_b0,r11__Z1 
 sv_function_helpervar_b0,r11__Z2  aux_cap_helper__b3,r5,band_UNPAIRED__Z2 
 sv_function_helpervar_b0,r11__Z3  aux_cap_helper__b9,r3,band_UNPAIRED__Z2 
 aux_cap_helper__b6,r4,band_LOW_PAIRED__Z1 
 aux_cap_helper__b6,r4,band_LOW_PAIRED__Z2  sv_function_helpervar_b2,r2__Z3 
 aux_cap_helper__b0,r11,band_UNPAIRED__Z1  sv_function_helpervar_b2,r2__Z1 
 aux_cap_helper__b0,r11,band_UNPAIRED__Z2  sv_function_helpervar_b2,r2__Z2 
 aux_cap_helper__b9,r7,band_UNPAIRED__Z1 
 aux_cap_helper__b9,r7,band_UNPAIRED__Z2 
 aux_cap_helper__b3,r9,band_UNPAIRED__Z1  sv_function_helpervar_b0,r5__Z1 
 aux_cap_helper__b3,r9,band_UNPAIRED__Z2  sv_function_helpervar_b0,r5__Z2 
 sv_function_helpervar_b0,r5__Z3  sv_function_helpervar_b7,r3__Z1 
 sv_function_helpervar_b7,r3__Z2  aux_cap_helper__b1,r8,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b1,r8,band_HIGH_PAIRED__Z1  sv_function_helpervar_b8,r10__Z1 
 aux_cap_helper__b7,r0,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b7,r0,band_HIGH_PAIRED__Z2  sv_function_helpervar_b8,r10__Z2 
 aux_cap_helper__b1,r8,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b7,r0,band_HIGH_PAIRED__Z1  sv_function_helpervar_b8,r10__Z3 
 sv_function_helpervar_b7,r3__Z3  aux_cap_helper__b7,r5,band_UNPAIRED__Z2 
 aux_cap_helper__b7,r5,band_UNPAIRED__Z1 
 aux_cap_helper__b8,r7,band_UNPAIRED__Z1 
 aux_cap_helper__b8,r7,band_UNPAIRED__Z2 
 aux_cap_helper__b2,r9,band_UNPAIRED__Z1 
 aux_cap_helper__b2,r9,band_UNPAIRED__Z2 
 aux_cap_helper__b9,r13,band_UNPAIRED__Z1 
 aux_cap_helper__b9,r13,band_UNPAIRED__Z2 
 aux_cap_helper__b7,r1,band_UNPAIRED__Z1 
 aux_cap_helper__b7,r9,band_LOW_PAIRED__Z2 
 aux_cap_helper__b7,r9,band_LOW_PAIRED__Z1 
 aux_cap_helper__b1,r1,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b1,r1,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b1,r1,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b7,r1,band_UNPAIRED__Z2 
 aux_cap_helper__b4,r12,band_UNPAIRED__Z1 
 aux_cap_helper__b4,r12,band_UNPAIRED__Z2 
 aux_cap_helper__b6,r0,band_LOW_PAIRED__Z1 
 aux_cap_helper__b6,r0,band_LOW_PAIRED__Z2 
 aux_cap_helper__b2,r5,band_UNPAIRED__Z2 
 aux_cap_helper__b2,r5,band_UNPAIRED__Z1 
 aux_cap_helper__b8,r3,band_UNPAIRED__Z1 
 aux_cap_helper__b8,r3,band_UNPAIRED__Z2 
 aux_cap_helper__b1,r3,band_UNPAIRED__Z1 
 aux_cap_helper__b5,r4,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b5,r4,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b5,r4,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b8,r12,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b8,r12,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b5,r3,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b5,r3,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b8,r12,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b5,r3,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b1,r3,band_UNPAIRED__Z2  sv_function_helpervar_b5,r2__Z3 
 sv_function_helpervar_b5,r2__Z2  sv_function_helpervar_b5,r2__Z1 
 aux_cap_helper__b2,r9,band_LOW_PAIRED__Z2 
 aux_cap_helper__b2,r9,band_LOW_PAIRED__Z1 
 aux_cap_helper__b0,r3,band_UNPAIRED__Z2 
 aux_cap_helper__b0,r3,band_UNPAIRED__Z1 
 aux_cap_helper__b5,r10,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b5,r10,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b5,r10,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b5,r8,band_LOW_PAIRED__Z2 
 aux_cap_helper__b5,r8,band_LOW_PAIRED__Z1  sv_function_helpervar_b6,r13__Z3 
 sv_function_helpervar_b6,r13__Z2  sv_function_helpervar_b6,r13__Z1 
 aux_cap_helper__b2,r10,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b2,r10,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b2,r10,band_HIGH_PAIRED__Z3  sv_function_helpervar_b4,r7__Z1 
 aux_cap_helper__b1,r0,band_HIGH_PAIRED__Z3  sv_function_helpervar_b4,r7__Z3 
 sv_function_helpervar_b4,r7__Z2  aux_cap_helper__b1,r0,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b1,r0,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b0,r4,band_LOW_PAIRED__Z2 
 aux_cap_helper__b0,r4,band_LOW_PAIRED__Z1 
 aux_cap_helper__b3,r5,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b3,r5,band_HIGH_PAIRED__Z3  sv_function_helpervar_b0,r1__Z1 
 aux_cap_helper__b2,r5,band_LOW_PAIRED__Z2 
 aux_cap_helper__b2,r5,band_LOW_PAIRED__Z1 
 aux_cap_helper__b3,r5,band_HIGH_PAIRED__Z1  sv_function_helpervar_b0,r1__Z3 
 sv_function_helpervar_b0,r1__Z2  aux_cap_helper__b6,r9,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b6,r9,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b6,r1,band_UNPAIRED__Z1 
 aux_cap_helper__b6,r1,band_UNPAIRED__Z2  sv_function_helpervar_b9,r8__Z1 
 aux_cap_helper__b6,r9,band_HIGH_PAIRED__Z3  sv_function_helpervar_b9,r8__Z2 
 sv_function_helpervar_b9,r8__Z3  aux_cap_helper__b9,r3,band_LOW_PAIRED__Z1 
 aux_cap_helper__b9,r3,band_LOW_PAIRED__Z2 
 aux_cap_helper__b0,r0,band_LOW_PAIRED__Z2 
 aux_cap_helper__b4,r11,band_LOW_PAIRED__Z2 
 aux_cap_helper__b4,r11,band_LOW_PAIRED__Z1 
 aux_cap_helper__b0,r0,band_LOW_PAIRED__Z1 
 aux_cap_helper__b8,r11,band_LOW_PAIRED__Z2 
 aux_cap_helper__b8,r11,band_LOW_PAIRED__Z1  sv_function_helpervar_b7,r0__Z3 
 sv_function_helpervar_b7,r0__Z2  sv_function_helpervar_b7,r0__Z1 
 aux_cap_helper__b2,r13,band_UNPAIRED__Z1 
 aux_cap_helper__b2,r13,band_UNPAIRED__Z2  sv_function_helpervar_b0,r9__Z2 
 sv_function_helpervar_b1,r4__Z3  sv_function_helpervar_b0,r9__Z3 
 sv_function_helpervar_b1,r4__Z2  sv_function_helpervar_b1,r4__Z1 
 aux_cap_helper__b6,r13,band_LOW_PAIRED__Z2 
 aux_cap_helper__b6,r13,band_LOW_PAIRED__Z1  sv_function_helpervar_b0,r9__Z1 
 sv_function_helpervar_b8,r13__Z1  sv_function_helpervar_b8,r13__Z2 
 sv_function_helpervar_b8,r13__Z3  aux_cap_helper__b3,r3,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b3,r3,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b3,r3,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b0,r1,band_UNPAIRED__Z1 
 aux_cap_helper__b0,r1,band_UNPAIRED__Z2  sv_function_helpervar_b6,r5__Z3 
 sv_function_helpervar_b6,r5__Z2  sv_function_helpervar_b6,r5__Z1 
 sv_function_helpervar_b9,r5__Z3  aux_cap_helper__b6,r10,band_UNPAIRED__Z1 
 aux_cap_helper__b6,r10,band_UNPAIRED__Z2 
 aux_cap_helper__b8,r8,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b8,r8,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b8,r8,band_HIGH_PAIRED__Z2  sv_function_helpervar_b9,r5__Z1 
 sv_function_helpervar_b9,r5__Z2  aux_cap_helper__b3,r1,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b3,r1,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b3,r1,band_HIGH_PAIRED__Z2  sv_function_helpervar_b3,r9__Z1 
 sv_function_helpervar_b3,r9__Z2  sv_function_helpervar_b3,r9__Z3 
 aux_cap_helper__b8,r12,band_LOW_PAIRED__Z1 
 aux_cap_helper__b8,r12,band_LOW_PAIRED__Z2  sv_function_helpervar_b4,r4__Z1 
 sv_function_helpervar_b4,r4__Z2  sv_function_helpervar_b4,r4__Z3 
 aux_cap_helper__b5,r12,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b5,r12,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b5,r1,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b5,r1,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b5,r1,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b8,r10,band_LOW_PAIRED__Z1 
 aux_cap_helper__b8,r10,band_LOW_PAIRED__Z2 
 aux_cap_helper__b8,r13,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b2,r11,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b2,r11,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b2,r11,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b4,r1,band_UNPAIRED__Z2 
 aux_cap_helper__b8,r13,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b4,r1,band_UNPAIRED__Z1 
 aux_cap_helper__b8,r13,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b5,r12,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b6,r11,band_LOW_PAIRED__Z1 
 aux_cap_helper__b6,r11,band_LOW_PAIRED__Z2 
 aux_cap_helper__b1,r2,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b1,r2,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b1,r2,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b5,r12,band_UNPAIRED__Z2 
 aux_cap_helper__b5,r12,band_UNPAIRED__Z1 
 aux_cap_helper__b6,r12,band_UNPAIRED__Z2 
 aux_cap_helper__b8,r7,band_LOW_PAIRED__Z2 
 aux_cap_helper__b6,r12,band_UNPAIRED__Z1 
 aux_cap_helper__b8,r7,band_LOW_PAIRED__Z1 
 aux_cap_helper__b2,r3,band_UNPAIRED__Z1 
 aux_cap_helper__b2,r3,band_UNPAIRED__Z2  sv_function_helpervar_b5,r12__Z2 
 sv_function_helpervar_b5,r12__Z1  sv_function_helpervar_b5,r12__Z3 
 aux_cap_helper__b4,r7,band_UNPAIRED__Z2 
 aux_cap_helper__b4,r7,band_UNPAIRED__Z1 
 aux_cap_helper__b6,r9,band_UNPAIRED__Z2 
 aux_cap_helper__b6,r9,band_UNPAIRED__Z1 
 aux_cap_helper__b7,r7,band_UNPAIRED__Z1 
 aux_cap_helper__b7,r7,band_UNPAIRED__Z2 
 aux_cap_helper__b5,r7,band_LOW_PAIRED__Z2 
 aux_cap_helper__b6,r7,band_UNPAIRED__Z2 
 aux_cap_helper__b6,r7,band_UNPAIRED__Z1 
 aux_cap_helper__b5,r7,band_LOW_PAIRED__Z1 
 aux_cap_helper__b0,r1,band_LOW_PAIRED__Z1 
 aux_cap_helper__b0,r1,band_LOW_PAIRED__Z2  sv_function_helpervar_b4,r12__Z1 
 sv_function_helpervar_b4,r12__Z2  aux_cap_helper__b7,r9,band_UNPAIRED__Z1 
 sv_function_helpervar_b4,r12__Z3  aux_cap_helper__b1,r5,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b1,r5,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b1,r5,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b7,r9,band_UNPAIRED__Z2  sv_function_helpervar_b1,r0__Z2 
 sv_function_helpervar_b1,r0__Z3  sv_function_helpervar_b1,r0__Z1 
 aux_cap_helper__b8,r9,band_UNPAIRED__Z2 
 aux_cap_helper__b8,r9,band_UNPAIRED__Z1 
 aux_cap_helper__b6,r5,band_UNPAIRED__Z1 
 aux_cap_helper__b6,r5,band_UNPAIRED__Z2  sv_function_helpervar_b6,r1__Z2 
 sv_function_helpervar_b6,r1__Z3  sv_function_helpervar_b6,r1__Z1 
 aux_cap_helper__b4,r5,band_UNPAIRED__Z1 
 aux_cap_helper__b4,r5,band_UNPAIRED__Z2 
 aux_cap_helper__b5,r3,band_UNPAIRED__Z2 
 aux_cap_helper__b5,r3,band_UNPAIRED__Z1  sv_function_helpervar_b9,r1__Z3 
 sv_function_helpervar_b9,r1__Z1  sv_function_helpervar_b9,r1__Z2 
 aux_cap_helper__b3,r1,band_LOW_PAIRED__Z1 
 aux_cap_helper__b3,r1,band_LOW_PAIRED__Z2 
 aux_cap_helper__b5,r7,band_UNPAIRED__Z1 
 aux_cap_helper__b1,r3,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b1,r3,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b1,r3,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b3,r3,band_UNPAIRED__Z1 
 aux_cap_helper__b5,r7,band_UNPAIRED__Z2 
 aux_cap_helper__b3,r3,band_UNPAIRED__Z2 
 aux_cap_helper__b1,r11,band_UNPAIRED__Z2 
 aux_cap_helper__b1,r11,band_UNPAIRED__Z1  sv_function_helpervar_b4,r0__Z3 
 aux_cap_helper__b5,r6,band_LOW_PAIRED__Z2 
 aux_cap_helper__b5,r6,band_LOW_PAIRED__Z1  GlobalBidder_W_ir__b7,r3 
 GlobalBidder_W_ir__b7,r2  GlobalBidder_W_ir__b7,r1  GlobalBidder_W_ir__b7,r0 
 aux_cap_helper__b2,r1,band_UNPAIRED__Z2 
 aux_cap_helper__b2,r1,band_UNPAIRED__Z1  sv_function_helpervar_b9,r10__Z3 
 sv_function_helpervar_b9,r10__Z1  sv_function_helpervar_b9,r10__Z2 
 aux_cap_helper__b3,r1,band_UNPAIRED__Z2 
 aux_cap_helper__b3,r1,band_UNPAIRED__Z1 
 aux_cap_helper__b0,r2,band_LOW_PAIRED__Z1 
 aux_cap_helper__b0,r2,band_LOW_PAIRED__Z2  sv_function_helpervar_b3,r5__Z3 
 sv_function_helpervar_b3,r5__Z2  sv_function_helpervar_b3,r5__Z1 
 sv_function_helpervar_b2,r11__Z3  aux_cap_helper__b5,r10,band_UNPAIRED__Z2 
 aux_cap_helper__b5,r10,band_UNPAIRED__Z1 
 aux_cap_helper__b6,r1,band_LOW_PAIRED__Z1 
 aux_cap_helper__b6,r1,band_LOW_PAIRED__Z2 
 aux_cap_helper__b6,r12,band_LOW_PAIRED__Z2 
 aux_cap_helper__b6,r12,band_LOW_PAIRED__Z1  GlobalBidder_W_ir__b7,r7 
 GlobalBidder_W_ir__b7,r6  sv_function_helpervar_b2,r11__Z2 
 GlobalBidder_W_ir__b7,r5  sv_function_helpervar_b2,r11__Z1 
 GlobalBidder_W_ir__b7,r4  GlobalBidder_W_ir__b7,r9  GlobalBidder_W_ir__b7,r8 
 aux_cap_helper__b2,r7,band_LOW_PAIRED__Z2 
 aux_cap_helper__b2,r7,band_LOW_PAIRED__Z1  sv_function_helpervar_b4,r0__Z1 
 sv_function_helpervar_b4,r0__Z2  aux_cap_helper__b8,r6,band_LOW_PAIRED__Z2 
 aux_cap_helper__b8,r6,band_LOW_PAIRED__Z1 
 aux_cap_helper__b5,r0,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b5,r0,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b4,r3,band_UNPAIRED__Z1 
 aux_cap_helper__b4,r3,band_UNPAIRED__Z2 
 aux_cap_helper__b2,r13,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b2,r13,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b5,r5,band_UNPAIRED__Z1 
 aux_cap_helper__b5,r5,band_UNPAIRED__Z2 
 aux_cap_helper__b2,r13,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b5,r0,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b1,r4,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b1,r4,band_HIGH_PAIRED__Z2  sv_function_helpervar_b7,r13__Z1 
 aux_cap_helper__b1,r13,band_UNPAIRED__Z2  sv_function_helpervar_b7,r13__Z2 
 aux_cap_helper__b1,r13,band_UNPAIRED__Z1  sv_function_helpervar_b7,r13__Z3 
 aux_cap_helper__b1,r4,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b3,r2,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b3,r2,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b4,r13,band_LOW_PAIRED__Z2 
 aux_cap_helper__b3,r2,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b4,r13,band_LOW_PAIRED__Z1  sv_function_helpervar_b8,r6__Z3 
 sv_function_helpervar_b8,r6__Z2  sv_function_helpervar_b8,r6__Z1 
 aux_cap_helper__b9,r1,band_LOW_PAIRED__Z1 
 aux_cap_helper__b9,r1,band_LOW_PAIRED__Z2  sv_function_helpervar_b3,r0__Z3 
 sv_function_helpervar_b3,r0__Z2  aux_cap_helper__b9,r1,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b9,r1,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b9,r1,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b0,r3,band_LOW_PAIRED__Z1 
 aux_cap_helper__b0,r3,band_LOW_PAIRED__Z2  sv_function_helpervar_b0,r2__Z2 
 sv_function_helpervar_b0,r2__Z1  sv_function_helpervar_b0,r2__Z3 
 GlobalBidder_W_ir__b9,r6  GlobalBidder_W_ir__b9,r7  GlobalBidder_W_ir__b9,r8 
 GlobalBidder_W_ir__b9,r9  sv_function_helpervar_b3,r0__Z1 
 aux_cap_helper__b8,r4,band_LOW_PAIRED__Z2 
 aux_cap_helper__b8,r4,band_LOW_PAIRED__Z1 
 aux_cap_helper__b2,r6,band_LOW_PAIRED__Z2 
 aux_cap_helper__b2,r6,band_LOW_PAIRED__Z1 
 aux_cap_helper__b6,r11,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b6,r11,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b8,r3,band_HIGH_PAIRED__Z3  sv_function_helpervar_b6,r12__Z3 
 sv_function_helpervar_b6,r12__Z2  aux_cap_helper__b6,r11,band_HIGH_PAIRED__Z1 
 sv_function_helpervar_b6,r12__Z1  aux_cap_helper__b8,r3,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b8,r3,band_HIGH_PAIRED__Z2  sv_function_helpervar_b8,r3__Z1 
 sv_function_helpervar_b8,r3__Z3  sv_function_helpervar_b8,r3__Z2 
 sv_function_helpervar_b7,r6__Z2  sv_function_helpervar_b7,r6__Z3 
 aux_cap_helper__b9,r9,band_LOW_PAIRED__Z2  sv_function_helpervar_b7,r6__Z1 
 aux_cap_helper__b9,r10,band_HIGH_PAIRED__Z2  sv_function_helpervar_b2,r7__Z1 
 aux_cap_helper__b9,r10,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b9,r10,band_HIGH_PAIRED__Z1  sv_function_helpervar_b2,r7__Z2 
 sv_function_helpervar_b2,r7__Z3  aux_cap_helper__b4,r0,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b4,r0,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b4,r0,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b1,r1,band_LOW_PAIRED__Z1 
 aux_cap_helper__b1,r10,band_UNPAIRED__Z1 
 aux_cap_helper__b1,r10,band_UNPAIRED__Z2 
 aux_cap_helper__b1,r1,band_LOW_PAIRED__Z2 
 aux_cap_helper__b0,r13,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b0,r13,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b0,r13,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b5,r1,band_UNPAIRED__Z2 
 aux_cap_helper__b5,r1,band_UNPAIRED__Z1 
 aux_cap_helper__b2,r4,band_UNPAIRED__Z2 
 aux_cap_helper__b2,r4,band_UNPAIRED__Z1  sv_function_helpervar_b0,r12__Z2 
 sv_function_helpervar_b0,r12__Z3  sv_function_helpervar_b0,r12__Z1 
 aux_cap_helper__b4,r9,band_LOW_PAIRED__Z1 
 aux_cap_helper__b4,r9,band_LOW_PAIRED__Z2 
 aux_cap_helper__b6,r11,band_UNPAIRED__Z1 
 aux_cap_helper__b6,r11,band_UNPAIRED__Z2  sv_function_helpervar_b4,r13__Z1 
 sv_function_helpervar_b4,r13__Z2  sv_function_helpervar_b4,r13__Z3 
 sv_function_helpervar_b5,r5__Z1  sv_function_helpervar_b5,r5__Z2 
 aux_cap_helper__b4,r12,band_LOW_PAIRED__Z2 
 aux_cap_helper__b4,r12,band_LOW_PAIRED__Z1  sv_function_helpervar_b5,r5__Z3 
 aux_cap_helper__b9,r2,band_LOW_PAIRED__Z2 
 aux_cap_helper__b9,r2,band_LOW_PAIRED__Z1 
 aux_cap_helper__b0,r6,band_UNPAIRED__Z1 
 aux_cap_helper__b0,r6,band_UNPAIRED__Z2 
 aux_cap_helper__b6,r4,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b6,r4,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b6,r4,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b9,r5,band_UNPAIRED__Z2 
 aux_cap_helper__b9,r5,band_UNPAIRED__Z1  sv_function_helpervar_b6,r9__Z3 
 sv_function_helpervar_b6,r9__Z2  sv_function_helpervar_b6,r9__Z1 
 aux_cap_helper__b1,r6,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b1,r6,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b1,r6,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b5,r9,band_UNPAIRED__Z2 
 aux_cap_helper__b5,r9,band_UNPAIRED__Z1 
 aux_cap_helper__b8,r6,band_UNPAIRED__Z1 
 aux_cap_helper__b8,r6,band_UNPAIRED__Z2  sv_function_helpervar_b9,r7__Z3 
 sv_function_helpervar_b9,r7__Z1  sv_function_helpervar_b9,r7__Z2 
 aux_cap_helper__b4,r11,band_UNPAIRED__Z2 
 aux_cap_helper__b4,r11,band_UNPAIRED__Z1 
 aux_cap_helper__b3,r0,band_LOW_PAIRED__Z1 
 aux_cap_helper__b3,r0,band_LOW_PAIRED__Z2 
 aux_cap_helper__b3,r6,band_UNPAIRED__Z2 
 aux_cap_helper__b3,r6,band_UNPAIRED__Z1 
 aux_cap_helper__b5,r9,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b5,r9,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b5,r9,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b7,r2,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b6,r3,band_UNPAIRED__Z2 
 aux_cap_helper__b6,r3,band_UNPAIRED__Z1 
 aux_cap_helper__b3,r4,band_LOW_PAIRED__Z1 
 aux_cap_helper__b7,r2,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b3,r4,band_LOW_PAIRED__Z2 
 aux_cap_helper__b7,r2,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b7,r10,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b7,r10,band_HIGH_PAIRED__Z2  sv_function_helpervar_b2,r3__Z1 
 aux_cap_helper__b7,r10,band_HIGH_PAIRED__Z3  sv_function_helpervar_b2,r3__Z2 
 sv_function_helpervar_b2,r3__Z3  aux_cap_helper__b7,r6,band_LOW_PAIRED__Z1 
 aux_cap_helper__b7,r6,band_LOW_PAIRED__Z2  sv_function_helpervar_b1,r11__Z1 
 sv_function_helpervar_b1,r11__Z3  sv_function_helpervar_b1,r11__Z2 
 aux_cap_helper__b9,r8,band_UNPAIRED__Z1 
 aux_cap_helper__b9,r8,band_UNPAIRED__Z2 
 aux_cap_helper__b5,r11,band_LOW_PAIRED__Z1 
 aux_cap_helper__b5,r11,band_LOW_PAIRED__Z2  sv_function_helpervar_b7,r2__Z2 
 sv_function_helpervar_b7,r2__Z3  sv_function_helpervar_b7,r2__Z1 
 aux_cap_helper__b8,r0,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b8,r0,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b8,r0,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b8,r11,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b8,r11,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b8,r11,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b6,r0,band_UNPAIRED__Z1 
 aux_cap_helper__b1,r5,band_UNPAIRED__Z1 
 aux_cap_helper__b3,r12,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b3,r12,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b6,r0,band_UNPAIRED__Z2 
 aux_cap_helper__b1,r5,band_UNPAIRED__Z2 
 aux_cap_helper__b3,r12,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b0,r8,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b0,r8,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b0,r8,band_HIGH_PAIRED__Z1  sv_function_helpervar_b3,r10__Z3 
 aux_cap_helper__b7,r2,band_UNPAIRED__Z1 
 aux_cap_helper__b7,r2,band_UNPAIRED__Z2 
 aux_cap_helper__b2,r7,band_UNPAIRED__Z1 
 aux_cap_helper__b2,r7,band_UNPAIRED__Z2  sv_function_helpervar_b5,r1__Z3 
 sv_function_helpervar_b8,r11__Z1  sv_function_helpervar_b8,r11__Z2 
 sv_function_helpervar_b5,r1__Z2  sv_function_helpervar_b8,r11__Z3 
 sv_function_helpervar_b5,r1__Z1  aux_cap_helper__b4,r2,band_LOW_PAIRED__Z1 
 sv_function_helpervar_b3,r10__Z1  aux_cap_helper__b4,r2,band_LOW_PAIRED__Z2 
 aux_cap_helper__b6,r10,band_LOW_PAIRED__Z2  sv_function_helpervar_b3,r10__Z2 
 aux_cap_helper__b6,r10,band_LOW_PAIRED__Z1 
 aux_cap_helper__b8,r8,band_LOW_PAIRED__Z1 
 aux_cap_helper__b8,r8,band_LOW_PAIRED__Z2 
 aux_cap_helper__b9,r12,band_UNPAIRED__Z2 
 aux_cap_helper__b9,r12,band_UNPAIRED__Z1  sv_function_helpervar_b4,r8__Z2 
 sv_function_helpervar_b4,r8__Z1  sv_function_helpervar_b4,r8__Z3 
 aux_cap_helper__b9,r4,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b9,r4,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b9,r4,band_HIGH_PAIRED__Z3  sv_function_helpervar_b6,r6__Z2 
 aux_cap_helper__b3,r7,band_LOW_PAIRED__Z2  sv_function_helpervar_b6,r6__Z1 
 aux_cap_helper__b3,r7,band_LOW_PAIRED__Z1  sv_function_helpervar_b6,r6__Z3 
 aux_cap_helper__b2,r11,band_UNPAIRED__Z1 
 aux_cap_helper__b2,r11,band_UNPAIRED__Z2  sv_function_helpervar_b5,r13__Z3 
 aux_cap_helper__b9,r6,band_LOW_PAIRED__Z2  sv_function_helpervar_b5,r13__Z2 
 aux_cap_helper__b9,r6,band_LOW_PAIRED__Z1  sv_function_helpervar_b5,r13__Z1 
 sv_function_helpervar_b9,r4__Z1  sv_function_helpervar_b9,r4__Z2 
 sv_function_helpervar_b9,r4__Z3  aux_cap_helper__b7,r12,band_UNPAIRED__Z2 
 aux_cap_helper__b7,r12,band_UNPAIRED__Z1  sv_function_helpervar_b4,r1__Z2 
 sv_function_helpervar_b4,r1__Z3  aux_cap_helper__b3,r13,band_UNPAIRED__Z2 
 aux_cap_helper__b3,r13,band_UNPAIRED__Z1 
 aux_cap_helper__b8,r6,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b8,r6,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b8,r6,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b4,r5,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b4,r5,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b4,r5,band_HIGH_PAIRED__Z2  sv_function_helpervar_b3,r8__Z1 
 sv_function_helpervar_b3,r8__Z3  sv_function_helpervar_b3,r8__Z2 
 aux_cap_helper__b0,r7,band_LOW_PAIRED__Z1 
 aux_cap_helper__b0,r7,band_LOW_PAIRED__Z2  sv_function_helpervar_b4,r1__Z1 
 aux_cap_helper__b2,r9,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b2,r9,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b2,r9,band_HIGH_PAIRED__Z3  sv_function_helpervar_b2,r0__Z1 
 sv_function_helpervar_b2,r0__Z2  sv_function_helpervar_b2,r0__Z3 
 aux_cap_helper__b8,r1,band_LOW_PAIRED__Z2 
 aux_cap_helper__b8,r1,band_LOW_PAIRED__Z1 
 aux_cap_helper__b1,r8,band_UNPAIRED__Z1 
 aux_cap_helper__b1,r8,band_UNPAIRED__Z2 
 aux_cap_helper__b7,r10,band_UNPAIRED__Z2 
 aux_cap_helper__b7,r10,band_UNPAIRED__Z1 
 aux_cap_helper__b0,r0,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b0,r0,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b0,r0,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b1,r9,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b2,r2,band_LOW_PAIRED__Z2 
 aux_cap_helper__b1,r9,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b1,r9,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b3,r7,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b3,r7,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b3,r7,band_HIGH_PAIRED__Z2  sv_function_helpervar_b0,r6__Z3 
 sv_function_helpervar_b0,r6__Z1  sv_function_helpervar_b0,r6__Z2 
 aux_cap_helper__b8,r1,band_UNPAIRED__Z1 
 aux_cap_helper__b8,r1,band_UNPAIRED__Z2 
 aux_cap_helper__b2,r2,band_LOW_PAIRED__Z1 
 aux_cap_helper__b4,r13,band_HIGH_PAIRED__Z1  sv_function_helpervar_b6,r2__Z3 
 sv_function_helpervar_b1,r7__Z2  sv_function_helpervar_b1,r7__Z1 
 sv_function_helpervar_b1,r7__Z3  aux_cap_helper__b4,r13,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b4,r13,band_HIGH_PAIRED__Z2  sv_function_helpervar_b9,r0__Z1 
 sv_function_helpervar_b9,r0__Z2  sv_function_helpervar_b9,r0__Z3 
 aux_cap_helper__b5,r3,band_LOW_PAIRED__Z2 
 aux_cap_helper__b5,r3,band_LOW_PAIRED__Z1 
 aux_cap_helper__b9,r2,band_UNPAIRED__Z1 
 aux_cap_helper__b9,r2,band_UNPAIRED__Z2 
 aux_cap_helper__b7,r12,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b7,r12,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b7,r12,band_HIGH_PAIRED__Z1  GlobalBidder_W_ir__b8,r10 
 GlobalBidder_W_ir__b8,r11  GlobalBidder_W_ir__b8,r12 
 GlobalBidder_W_ir__b8,r13  aux_cap_helper__b1,r12,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b1,r12,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b1,r12,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b4,r11,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b4,r11,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b4,r11,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b7,r13,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b7,r13,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b7,r13,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b6,r7,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b6,r7,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b6,r7,band_HIGH_PAIRED__Z2  sv_function_helpervar_b7,r12__Z3 
 aux_cap_helper__b2,r10,band_LOW_PAIRED__Z1  sv_function_helpervar_b7,r12__Z1 
 aux_cap_helper__b2,r10,band_LOW_PAIRED__Z2  sv_function_helpervar_b7,r12__Z2 
 aux_cap_helper__b8,r11,band_UNPAIRED__Z2 
 aux_cap_helper__b8,r11,band_UNPAIRED__Z1 
 aux_cap_helper__b0,r12,band_LOW_PAIRED__Z2 
 aux_cap_helper__b0,r12,band_LOW_PAIRED__Z1 
 aux_cap_helper__b0,r6,band_LOW_PAIRED__Z1 
 aux_cap_helper__b0,r6,band_LOW_PAIRED__Z2  sv_function_helpervar_b2,r10__Z3 
 sv_function_helpervar_b2,r10__Z2  sv_function_helpervar_b2,r10__Z1 
 aux_cap_helper__b5,r6,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b5,r6,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b5,r6,band_HIGH_PAIRED__Z3  sv_function_helpervar_b3,r4__Z1 
 sv_function_helpervar_b3,r4__Z3  sv_function_helpervar_b3,r4__Z2 
 sv_function_helpervar_b5,r9__Z1  sv_function_helpervar_b5,r9__Z2 
 sv_function_helpervar_b5,r9__Z3  aux_cap_helper__b3,r10,band_LOW_PAIRED__Z2 
 aux_cap_helper__b3,r10,band_LOW_PAIRED__Z1 
 aux_cap_helper__b6,r5,band_LOW_PAIRED__Z1 
 aux_cap_helper__b6,r5,band_LOW_PAIRED__Z2 
 aux_cap_helper__b7,r5,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b7,r5,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b7,r5,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b3,r12,band_UNPAIRED__Z1 
 aux_cap_helper__b3,r12,band_UNPAIRED__Z2 
 aux_cap_helper__b3,r10,band_UNPAIRED__Z2 
 aux_cap_helper__b3,r10,band_UNPAIRED__Z1  sv_function_helpervar_b8,r7__Z1 
 sv_function_helpervar_b8,r7__Z3  sv_function_helpervar_b8,r7__Z2 
 aux_cap_helper__b9,r5,band_LOW_PAIRED__Z2 
 aux_cap_helper__b9,r5,band_LOW_PAIRED__Z1 
 aux_cap_helper__b1,r11,band_LOW_PAIRED__Z2 
 aux_cap_helper__b1,r11,band_LOW_PAIRED__Z1 
 aux_cap_helper__b4,r8,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b4,r8,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b4,r8,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b1,r8,band_LOW_PAIRED__Z2 
 aux_cap_helper__b9,r0,band_UNPAIRED__Z2 
 aux_cap_helper__b1,r8,band_LOW_PAIRED__Z1 
 aux_cap_helper__b9,r0,band_UNPAIRED__Z1 
 aux_cap_helper__b0,r9,band_UNPAIRED__Z2 
 aux_cap_helper__b0,r9,band_UNPAIRED__Z1  GlobalBidder_hat{W}__i9,k2 
 GlobalBidder_W_ir__b9,r2  GlobalBidder_hat{W}__i9,k1 
 GlobalBidder_W_ir__b9,r3  GlobalBidder_W_ir__b9,r4  GlobalBidder_W_ir__b9,r5 
 GlobalBidder_hat{W}__i9,k0  GlobalBidder_W_ir__b9,r0 
 GlobalBidder_W_ir__b9,r1  aux_cap_helper__b2,r3,band_LOW_PAIRED__Z2 
 aux_cap_helper__b2,r3,band_LOW_PAIRED__Z1  sv_function_helpervar_b6,r2__Z1 
 sv_function_helpervar_b6,r2__Z2  sv_function_helpervar_b1,r3__Z2 
 sv_function_helpervar_b1,r3__Z1  sv_function_helpervar_b1,r3__Z3 
 aux_cap_helper__b8,r13,band_UNPAIRED__Z2 
 aux_cap_helper__b8,r13,band_UNPAIRED__Z1 
 aux_cap_helper__b4,r2,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b4,r2,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b4,r2,band_HIGH_PAIRED__Z2  sv_function_helpervar_b4,r9__Z1 
 sv_function_helpervar_b4,r9__Z3  sv_function_helpervar_b4,r9__Z2 
 aux_cap_helper__b9,r12,band_HIGH_PAIRED__Z2  sv_function_helpervar_b6,r11__Z3 
 aux_cap_helper__b9,r12,band_HIGH_PAIRED__Z3  sv_function_helpervar_b6,r11__Z2 
 sv_function_helpervar_b6,r11__Z1  aux_cap_helper__b9,r12,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b7,r13,band_UNPAIRED__Z2 
 aux_cap_helper__b7,r13,band_UNPAIRED__Z1  sv_function_helpervar_b9,r11__Z1 
 sv_function_helpervar_b9,r11__Z2  sv_function_helpervar_b9,r11__Z3 
 aux_cap_helper__b9,r3,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b9,r3,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b9,r3,band_HIGH_PAIRED__Z3  sv_function_helpervar_b5,r4__Z3 
 sv_function_helpervar_b5,r4__Z2  sv_function_helpervar_b5,r4__Z1 
 aux_cap_helper__b3,r10,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b3,r10,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b3,r10,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b6,r13,band_UNPAIRED__Z2 
 aux_cap_helper__b6,r13,band_UNPAIRED__Z1 
 aux_cap_helper__b7,r7,band_LOW_PAIRED__Z1  sv_function_helpervar_b0,r3__Z3 
 sv_function_helpervar_b0,r3__Z2  sv_function_helpervar_b0,r3__Z1 
 aux_cap_helper__b5,r10,band_LOW_PAIRED__Z1 
 aux_cap_helper__b5,r10,band_LOW_PAIRED__Z2 
 aux_cap_helper__b7,r7,band_LOW_PAIRED__Z2 
 aux_cap_helper__b8,r1,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b8,r1,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b8,r1,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b9,r7,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b9,r7,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b9,r7,band_HIGH_PAIRED__Z1  sv_function_helpervar_b7,r9__Z1 
 sv_function_helpervar_b7,r9__Z2  sv_function_helpervar_b7,r9__Z3 
 aux_cap_helper__b2,r2,band_UNPAIRED__Z1 
 aux_cap_helper__b2,r2,band_UNPAIRED__Z2 
 aux_cap_helper__b7,r7,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b7,r7,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b7,r7,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b0,r3,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b0,r3,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b0,r3,band_HIGH_PAIRED__Z2  sv_function_helpervar_b1,r9__Z3 
 sv_function_helpervar_b1,r9__Z2  sv_function_helpervar_b1,r9__Z1 
 aux_cap_helper__b0,r7,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b0,r7,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b0,r7,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b0,r0,band_UNPAIRED__Z2 
 aux_cap_helper__b0,r0,band_UNPAIRED__Z1 
 aux_cap_helper__b1,r12,band_LOW_PAIRED__Z1 
 aux_cap_helper__b1,r12,band_LOW_PAIRED__Z2  sv_function_helpervar_b2,r8__Z1 
 sv_function_helpervar_b2,r8__Z2  sv_function_helpervar_b2,r8__Z3 
 sv_function_helpervar_b8,r0__Z3  sv_function_helpervar_b8,r0__Z2 
 sv_function_helpervar_b8,r0__Z1  aux_cap_helper__b5,r13,band_UNPAIRED__Z1 
 aux_cap_helper__b5,r13,band_UNPAIRED__Z2 
 aux_cap_helper__b4,r1,band_LOW_PAIRED__Z2 
 aux_cap_helper__b4,r1,band_LOW_PAIRED__Z1 
 aux_cap_helper__b1,r7,band_LOW_PAIRED__Z1 
 aux_cap_helper__b1,r7,band_LOW_PAIRED__Z2 
 aux_cap_helper__b9,r13,band_LOW_PAIRED__Z2 
 aux_cap_helper__b9,r13,band_LOW_PAIRED__Z1 
 aux_cap_helper__b4,r6,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b4,r6,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b4,r6,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b4,r0,band_UNPAIRED__Z2 
 aux_cap_helper__b4,r0,band_UNPAIRED__Z1  sv_function_helpervar_b5,r10__Z1 
 sv_function_helpervar_b5,r10__Z2  sv_function_helpervar_b5,r10__Z3 
 sv_function_helpervar_b1,r12__Z3  aux_cap_helper__b1,r2,band_LOW_PAIRED__Z2 
 aux_cap_helper__b1,r2,band_LOW_PAIRED__Z1 
 aux_cap_helper__b4,r6,band_LOW_PAIRED__Z2 
 aux_cap_helper__b4,r6,band_LOW_PAIRED__Z1  sv_function_helpervar_b1,r12__Z2 
 sv_function_helpervar_b1,r12__Z1  aux_cap_helper__b8,r4,band_UNPAIRED__Z2 
 aux_cap_helper__b8,r4,band_UNPAIRED__Z1  sv_function_helpervar_b5,r0__Z3 
 sv_function_helpervar_b5,r0__Z2  sv_function_helpervar_b5,r0__Z1 
 aux_cap_helper__b9,r6,band_UNPAIRED__Z2 
 aux_cap_helper__b9,r6,band_UNPAIRED__Z1  sv_function_helpervar_b4,r5__Z1 
 sv_function_helpervar_b4,r5__Z2  sv_function_helpervar_b4,r5__Z3 
 aux_cap_helper__b7,r5,band_LOW_PAIRED__Z2 
 aux_cap_helper__b7,r5,band_LOW_PAIRED__Z1 
 aux_cap_helper__b4,r8,band_UNPAIRED__Z1 
 aux_cap_helper__b4,r8,band_UNPAIRED__Z2 
 aux_cap_helper__b7,r10,band_LOW_PAIRED__Z1 
 aux_cap_helper__b7,r10,band_LOW_PAIRED__Z2 
 aux_cap_helper__b6,r1,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b6,r1,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b6,r2,band_UNPAIRED__Z2 
 aux_cap_helper__b6,r2,band_UNPAIRED__Z1 
 aux_cap_helper__b6,r1,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b9,r10,band_UNPAIRED__Z1 
 aux_cap_helper__b9,r10,band_UNPAIRED__Z2 
 aux_cap_helper__b1,r6,band_UNPAIRED__Z1 
 aux_cap_helper__b1,r6,band_UNPAIRED__Z2 
 aux_cap_helper__b7,r4,band_UNPAIRED__Z2 
 aux_cap_helper__b7,r4,band_UNPAIRED__Z1 
 aux_cap_helper__b0,r12,band_UNPAIRED__Z2 
 aux_cap_helper__b0,r12,band_UNPAIRED__Z1  sv_function_helpervar_b3,r13__Z2 
 sv_function_helpervar_b3,r13__Z3  sv_function_helpervar_b3,r13__Z1 
 aux_cap_helper__b7,r2,band_LOW_PAIRED__Z1 
 aux_cap_helper__b3,r12,band_LOW_PAIRED__Z1 
 aux_cap_helper__b7,r2,band_LOW_PAIRED__Z2 
 aux_cap_helper__b0,r4,band_UNPAIRED__Z1 
 aux_cap_helper__b3,r12,band_LOW_PAIRED__Z2 
 aux_cap_helper__b0,r4,band_UNPAIRED__Z2 
 aux_cap_helper__b2,r7,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b2,r7,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b7,r13,band_LOW_PAIRED__Z1 
 aux_cap_helper__b2,r7,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b7,r13,band_LOW_PAIRED__Z2  sv_function_helpervar_b8,r12__Z2 
 sv_function_helpervar_b8,r12__Z3  sv_function_helpervar_b8,r12__Z1 
 aux_cap_helper__b3,r13,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b7,r8,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b3,r13,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b7,r8,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b3,r13,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b7,r8,band_HIGH_PAIRED__Z3  sv_function_helpervar_b2,r4__Z1 
 sv_function_helpervar_b2,r4__Z2  sv_function_helpervar_b2,r4__Z3 
 aux_cap_helper__b0,r2,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b0,r2,band_HIGH_PAIRED__Z3  sv_function_helpervar_b7,r5__Z1 
 sv_function_helpervar_b7,r5__Z2  sv_function_helpervar_b7,r5__Z3 
 aux_cap_helper__b0,r2,band_HIGH_PAIRED__Z1  sv_function_helpervar_b9,r6__Z2 
 sv_function_helpervar_b9,r6__Z3  sv_function_helpervar_b9,r6__Z1 
 aux_cap_helper__b2,r6,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b0,r11,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b2,r6,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b0,r11,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b2,r6,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b0,r11,band_HIGH_PAIRED__Z3  GlobalBidder_W_ir__b8,r8 
 GlobalBidder_W_ir__b8,r7  GlobalBidder_W_ir__b8,r6  GlobalBidder_W_ir__b8,r5 
 GlobalBidder_W_ir__b8,r9  aux_cap_helper__b9,r6,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b6,r13,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b6,r13,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b9,r6,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b6,r13,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b9,r6,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b6,r2,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b6,r2,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b6,r2,band_HIGH_PAIRED__Z1  GlobalBidder_W_ir__b8,r0 
 GlobalBidder_W_ir__b8,r4  GlobalBidder_W_ir__b8,r3  GlobalBidder_W_ir__b8,r2 
 GlobalBidder_W_ir__b8,r1  aux_cap_helper__b3,r8,band_UNPAIRED__Z1 
 aux_cap_helper__b3,r8,band_UNPAIRED__Z2 
 aux_cap_helper__b2,r6,band_UNPAIRED__Z1 
 aux_cap_helper__b2,r6,band_UNPAIRED__Z2 
 aux_cap_helper__b6,r7,band_LOW_PAIRED__Z1 
 aux_cap_helper__b6,r7,band_LOW_PAIRED__Z2 
 aux_cap_helper__b2,r12,band_UNPAIRED__Z2 
 aux_cap_helper__b2,r12,band_UNPAIRED__Z1 
 aux_cap_helper__b9,r10,band_LOW_PAIRED__Z2 
 aux_cap_helper__b9,r10,band_LOW_PAIRED__Z1  sv_function_helpervar_b4,r2__Z3 
 sv_function_helpervar_b4,r2__Z1  sv_function_helpervar_b4,r2__Z2 
 sv_function_helpervar_b3,r7__Z2  sv_function_helpervar_b3,r7__Z1 
 sv_function_helpervar_b2,r13__Z2  sv_function_helpervar_b2,r13__Z1 
 sv_function_helpervar_b2,r13__Z3  sv_function_helpervar_b3,r7__Z3 
 aux_cap_helper__b4,r1,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b4,r1,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b4,r1,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b9,r11,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b9,r11,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b9,r11,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b1,r4,band_LOW_PAIRED__Z2 
 aux_cap_helper__b1,r4,band_LOW_PAIRED__Z1 
 aux_cap_helper__b7,r12,band_LOW_PAIRED__Z2 
 aux_cap_helper__b7,r12,band_LOW_PAIRED__Z1 
 aux_cap_helper__b4,r3,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b4,r3,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b4,r3,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b9,r13,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b9,r13,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b9,r13,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b7,r3,band_LOW_PAIRED__Z1 
 aux_cap_helper__b7,r3,band_LOW_PAIRED__Z2 
 aux_cap_helper__b5,r13,band_LOW_PAIRED__Z1 
 aux_cap_helper__b5,r13,band_LOW_PAIRED__Z2 
 aux_cap_helper__b0,r8,band_LOW_PAIRED__Z1 
 aux_cap_helper__b0,r8,band_LOW_PAIRED__Z2 
 aux_cap_helper__b8,r0,band_LOW_PAIRED__Z2 
 aux_cap_helper__b8,r0,band_LOW_PAIRED__Z1  sv_function_helpervar_b8,r8__Z3 
 sv_function_helpervar_b8,r8__Z2  sv_function_helpervar_b8,r8__Z1 
 aux_cap_helper__b1,r2,band_UNPAIRED__Z1 
 aux_cap_helper__b9,r11,band_LOW_PAIRED__Z2 
 aux_cap_helper__b1,r2,band_UNPAIRED__Z2 
 aux_cap_helper__b9,r11,band_LOW_PAIRED__Z1 
 aux_cap_helper__b0,r4,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b0,r4,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b0,r4,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b2,r0,band_LOW_PAIRED__Z2 
 aux_cap_helper__b2,r0,band_LOW_PAIRED__Z1 
 aux_cap_helper__b1,r3,band_LOW_PAIRED__Z1 
 aux_cap_helper__b1,r3,band_LOW_PAIRED__Z2 
 aux_cap_helper__b1,r0,band_UNPAIRED__Z2 
 aux_cap_helper__b1,r0,band_UNPAIRED__Z1 
 aux_cap_helper__b0,r10,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b0,r10,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b2,r5,band_HIGH_PAIRED__Z3  sv_function_helpervar_b1,r6__Z1 
 sv_function_helpervar_b1,r6__Z3  sv_function_helpervar_b1,r6__Z2 
 aux_cap_helper__b2,r5,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b2,r5,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b0,r10,band_HIGH_PAIRED__Z3  sv_function_helpervar_b0,r7__Z1 
 aux_cap_helper__b1,r5,band_LOW_PAIRED__Z1  sv_function_helpervar_b0,r7__Z2 
 aux_cap_helper__b3,r11,band_HIGH_PAIRED__Z1  sv_function_helpervar_b0,r7__Z3 
 aux_cap_helper__b3,r11,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b3,r11,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b1,r5,band_LOW_PAIRED__Z2 
 aux_cap_helper__b3,r4,band_UNPAIRED__Z1 
 aux_cap_helper__b3,r4,band_UNPAIRED__Z2 
 aux_cap_helper__b6,r12,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b3,r8,band_LOW_PAIRED__Z1 
 aux_cap_helper__b3,r8,band_LOW_PAIRED__Z2 
 aux_cap_helper__b6,r12,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b6,r12,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b5,r0,band_LOW_PAIRED__Z2 
 aux_cap_helper__b5,r0,band_LOW_PAIRED__Z1  sv_function_helpervar_b2,r1__Z2 
 sv_function_helpervar_b2,r1__Z3  sv_function_helpervar_b2,r1__Z1 
 aux_cap_helper__b4,r3,band_LOW_PAIRED__Z2 
 aux_cap_helper__b4,r3,band_LOW_PAIRED__Z1 
 aux_cap_helper__b2,r1,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b2,r1,band_HIGH_PAIRED__Z1  sv_function_helpervar_b9,r3__Z3 
 aux_cap_helper__b4,r5,band_LOW_PAIRED__Z2  sv_function_helpervar_b9,r3__Z1 
 aux_cap_helper__b2,r1,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b4,r5,band_LOW_PAIRED__Z1  sv_function_helpervar_b9,r3__Z2 
 aux_cap_helper__b9,r8,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b9,r8,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b9,r8,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b6,r8,band_LOW_PAIRED__Z1 
 aux_cap_helper__b6,r8,band_LOW_PAIRED__Z2  sv_function_helpervar_b6,r7__Z3 
 sv_function_helpervar_b6,r7__Z2  sv_function_helpervar_b6,r7__Z1 
 aux_cap_helper__b7,r6,band_UNPAIRED__Z1 
 aux_cap_helper__b7,r6,band_UNPAIRED__Z2 
 aux_cap_helper__b8,r8,band_UNPAIRED__Z1 
 aux_cap_helper__b8,r8,band_UNPAIRED__Z2 
 aux_cap_helper__b5,r2,band_UNPAIRED__Z2 
 aux_cap_helper__b5,r2,band_UNPAIRED__Z1 
 aux_cap_helper__b6,r8,band_UNPAIRED__Z2 
 aux_cap_helper__b3,r9,band_LOW_PAIRED__Z1 
 aux_cap_helper__b6,r8,band_UNPAIRED__Z1 
 aux_cap_helper__b3,r9,band_LOW_PAIRED__Z2 
 aux_cap_helper__b7,r8,band_UNPAIRED__Z2 
 aux_cap_helper__b7,r8,band_UNPAIRED__Z1 
 aux_cap_helper__b0,r10,band_UNPAIRED__Z2 
 aux_cap_helper__b0,r10,band_UNPAIRED__Z1 
 aux_cap_helper__b9,r8,band_LOW_PAIRED__Z2 
 aux_cap_helper__b9,r8,band_LOW_PAIRED__Z1 
 aux_cap_helper__b5,r6,band_UNPAIRED__Z1 
 aux_cap_helper__b5,r6,band_UNPAIRED__Z2 
 aux_cap_helper__b5,r8,band_UNPAIRED__Z2 
 aux_cap_helper__b5,r8,band_UNPAIRED__Z1  sv_function_helpervar_b3,r3__Z2 
 sv_function_helpervar_b3,r3__Z1  aux_cap_helper__b4,r6,band_UNPAIRED__Z2 
 sv_function_helpervar_b3,r3__Z3  aux_cap_helper__b4,r6,band_UNPAIRED__Z1 
 aux_cap_helper__b5,r4,band_UNPAIRED__Z1 
 aux_cap_helper__b5,r4,band_UNPAIRED__Z2 
 aux_cap_helper__b6,r0,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b6,r0,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b6,r0,band_HIGH_PAIRED__Z1  GlobalBidder_W_ir__b7,r13 
 GlobalBidder_W_ir__b7,r11  GlobalBidder_W_ir__b7,r12 
 GlobalBidder_W_ir__b7,r10  sv_function_helpervar_b5,r8__Z1 
 sv_function_helpervar_b5,r8__Z2  aux_cap_helper__b2,r4,band_HIGH_PAIRED__Z1 
 sv_function_helpervar_b5,r8__Z3  aux_cap_helper__b2,r4,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b5,r12,band_LOW_PAIRED__Z1 
 aux_cap_helper__b2,r4,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b5,r12,band_LOW_PAIRED__Z2 
 aux_cap_helper__b4,r2,band_UNPAIRED__Z2 
 aux_cap_helper__b4,r2,band_UNPAIRED__Z1 
 aux_cap_helper__b6,r6,band_UNPAIRED__Z1 
 aux_cap_helper__b6,r6,band_UNPAIRED__Z2 
 aux_cap_helper__b5,r11,band_UNPAIRED__Z2 
 aux_cap_helper__b5,r11,band_UNPAIRED__Z1 
 aux_cap_helper__b0,r6,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b0,r6,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b0,r6,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b2,r2,band_HIGH_PAIRED__Z1  sv_function_helpervar_b7,r11__Z1 
 aux_cap_helper__b2,r2,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b2,r2,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b0,r9,band_LOW_PAIRED__Z1 
 aux_cap_helper__b3,r13,band_LOW_PAIRED__Z2 
 aux_cap_helper__b0,r9,band_LOW_PAIRED__Z2 
 aux_cap_helper__b3,r13,band_LOW_PAIRED__Z1  sv_function_helpervar_b7,r11__Z3 
 sv_function_helpervar_b7,r11__Z2  GlobalBidder_hat{W}__i7,k0 
 GlobalBidder_hat{W}__i7,k1  GlobalBidder_hat{W}__i7,k2 
 sv_function_helpervar_b1,r2__Z1  sv_function_helpervar_b1,r2__Z3 
 aux_cap_helper__b3,r0,band_UNPAIRED__Z2  sv_function_helpervar_b1,r2__Z2 
 aux_cap_helper__b3,r0,band_UNPAIRED__Z1 
 aux_cap_helper__b2,r0,band_UNPAIRED__Z2 
 aux_cap_helper__b2,r0,band_UNPAIRED__Z1 
 aux_cap_helper__b3,r2,band_UNPAIRED__Z2 
 aux_cap_helper__b3,r2,band_UNPAIRED__Z1 
 aux_cap_helper__b9,r9,band_LOW_PAIRED__Z1  sv_function_helpervar_b8,r4__Z3 
 sv_function_helpervar_b8,r4__Z2  sv_function_helpervar_b8,r4__Z1 
 sv_function_helpervar_b4,r10__Z2  aux_cap_helper__b9,r9,band_HIGH_PAIRED__Z3 
 sv_function_helpervar_b4,r10__Z3  aux_cap_helper__b9,r9,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b9,r9,band_HIGH_PAIRED__Z1  sv_function_helpervar_b6,r3__Z1 
 aux_cap_helper__b7,r4,band_LOW_PAIRED__Z1 
 aux_cap_helper__b7,r4,band_LOW_PAIRED__Z2 
 aux_cap_helper__b7,r11,band_LOW_PAIRED__Z2  sv_function_helpervar_b4,r10__Z1 
 aux_cap_helper__b7,r11,band_LOW_PAIRED__Z1 
 aux_cap_helper__b1,r12,band_UNPAIRED__Z1 
 aux_cap_helper__b1,r12,band_UNPAIRED__Z2 
 aux_cap_helper__b2,r3,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b2,r3,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b2,r3,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b4,r4,band_UNPAIRED__Z1 
 aux_cap_helper__b4,r4,band_UNPAIRED__Z2 
 aux_cap_helper__b6,r10,band_HIGH_PAIRED__Z3 
 aux_cap_helper__b6,r10,band_HIGH_PAIRED__Z1  sv_function_helpervar_b6,r3__Z3 
 aux_cap_helper__b6,r10,band_HIGH_PAIRED__Z2  sv_function_helpervar_b6,r3__Z2 
 aux_cap_helper__b6,r9,band_LOW_PAIRED__Z2 
 aux_cap_helper__b6,r9,band_LOW_PAIRED__Z1 
 aux_cap_helper__b4,r4,band_LOW_PAIRED__Z1 
 aux_cap_helper__b4,r4,band_LOW_PAIRED__Z2 
 aux_cap_helper__b0,r5,band_HIGH_PAIRED__Z2 
 aux_cap_helper__b0,r5,band_HIGH_PAIRED__Z1 
 aux_cap_helper__b0,r5,band_HIGH_PAIRED__Z3 
Generals
 X__b2,r2,band_LOW_PAIRED  X__b2,r12,band_HIGH_PAIRED 
 X__b4,r7,band_LOW_PAIRED  X__b6,r0,band_LOW_PAIRED  X__b7,r1,band_HIGH_PAIRED 
 X__b5,r7,band_HIGH_PAIRED  X__b9,r12,band_LOW_PAIRED 
 X__b0,r9,band_LOW_PAIRED  X__b2,r9,band_HIGH_PAIRED  X__b4,r0,band_UNPAIRED 
 X__b8,r5,band_HIGH_PAIRED  X__b4,r13,band_LOW_PAIRED  X__b3,r10,band_UNPAIRED 
 X__b5,r1,band_LOW_PAIRED  X__b7,r11,band_UNPAIRED  X__b9,r8,band_HIGH_PAIRED 
 X__b7,r6,band_LOW_PAIRED  X__b5,r2,band_UNPAIRED  X__b1,r3,band_LOW_PAIRED 
 X__b4,r3,band_HIGH_PAIRED  X__b2,r12,band_LOW_PAIRED 
 X__b0,r2,band_HIGH_PAIRED  X__b5,r6,band_UNPAIRED  X__b8,r5,band_LOW_PAIRED 
 X__b5,r4,band_UNPAIRED  X__b7,r11,band_LOW_PAIRED  X__b5,r8,band_UNPAIRED 
 X__b5,r11,band_HIGH_PAIRED  X__b7,r13,band_HIGH_PAIRED 
 X__b7,r13,band_UNPAIRED  X__b0,r9,band_UNPAIRED  X__b5,r6,band_HIGH_PAIRED 
 X__b8,r4,band_HIGH_PAIRED  X__b2,r11,band_HIGH_PAIRED 
 X__b8,r10,band_HIGH_PAIRED  X__b6,r7,band_LOW_PAIRED  X__b4,r2,band_UNPAIRED 
 X__b1,r6,band_HIGH_PAIRED  X__b2,r8,band_HIGH_PAIRED 
 X__b0,r4,band_LOW_PAIRED  X__b2,r7,band_LOW_PAIRED  X__b7,r2,band_HIGH_PAIRED 
 X__b4,r4,band_UNPAIRED  X__b0,r11,band_LOW_PAIRED  X__b3,r0,band_UNPAIRED 
 X__b1,r5,band_HIGH_PAIRED  X__b4,r6,band_UNPAIRED  X__b0,r7,band_UNPAIRED 
 X__b5,r8,band_LOW_PAIRED  X__b4,r4,band_HIGH_PAIRED  X__b1,r8,band_LOW_PAIRED 
 X__b3,r2,band_UNPAIRED  X__b1,r1,band_LOW_PAIRED  X__b5,r12,band_HIGH_PAIRED 
 X__b7,r8,band_UNPAIRED  X__b3,r6,band_LOW_PAIRED  X__b7,r12,band_HIGH_PAIRED 
 X__b7,r4,band_LOW_PAIRED  X__b7,r1,band_LOW_PAIRED  X__b8,r10,band_LOW_PAIRED 
 X__b4,r5,band_HIGH_PAIRED  X__b0,r3,band_HIGH_PAIRED 
 X__b3,r11,band_LOW_PAIRED  X__b6,r2,band_LOW_PAIRED  X__b6,r5,band_LOW_PAIRED 
 X__b4,r2,band_HIGH_PAIRED  X__b0,r12,band_UNPAIRED  X__b9,r9,band_LOW_PAIRED 
 X__b2,r11,band_UNPAIRED  X__b0,r2,band_LOW_PAIRED  X__b4,r5,band_LOW_PAIRED 
 X__b4,r13,band_HIGH_PAIRED  X__b8,r3,band_HIGH_PAIRED 
 X__b8,r12,band_UNPAIRED  X__b6,r9,band_HIGH_PAIRED  X__b8,r6,band_HIGH_PAIRED 
 X__b6,r6,band_UNPAIRED  X__b0,r1,band_HIGH_PAIRED  X__b1,r3,band_HIGH_PAIRED 
 X__b5,r0,band_UNPAIRED  X__b1,r8,band_HIGH_PAIRED  X__b9,r5,band_UNPAIRED 
 X__b1,r10,band_LOW_PAIRED  X__b8,r0,band_LOW_PAIRED  X__b5,r6,band_LOW_PAIRED 
 X__b6,r12,band_UNPAIRED  X__b3,r12,band_UNPAIRED  X__b7,r4,band_UNPAIRED 
 X__b9,r10,band_LOW_PAIRED  X__b3,r0,band_HIGH_PAIRED 
 X__b5,r10,band_HIGH_PAIRED  X__b5,r9,band_HIGH_PAIRED 
 X__b2,r10,band_LOW_PAIRED  X__b2,r9,band_LOW_PAIRED 
 X__b2,r13,band_HIGH_PAIRED  X__b6,r2,band_UNPAIRED  X__b3,r8,band_LOW_PAIRED 
 X__b9,r1,band_UNPAIRED  X__b0,r11,band_HIGH_PAIRED 
 X__b8,r12,band_HIGH_PAIRED  X__b4,r0,band_HIGH_PAIRED  X__b8,r3,band_UNPAIRED 
 X__b4,r4,band_LOW_PAIRED  X__b6,r11,band_LOW_PAIRED 
 X__b9,r5,band_HIGH_PAIRED  X__b1,r1,band_UNPAIRED  X__b4,r12,band_HIGH_PAIRED 
 X__b6,r9,band_LOW_PAIRED  X__b1,r0,band_UNPAIRED  X__b1,r2,band_HIGH_PAIRED 
 X__b0,r6,band_LOW_PAIRED  X__b2,r6,band_HIGH_PAIRED  X__b1,r2,band_UNPAIRED 
 X__b0,r12,band_HIGH_PAIRED  X__b9,r9,band_UNPAIRED  X__b7,r3,band_LOW_PAIRED 
 X__b9,r10,band_UNPAIRED  X__b5,r4,band_LOW_PAIRED  X__b4,r6,band_HIGH_PAIRED 
 X__b7,r0,band_UNPAIRED  X__b9,r11,band_UNPAIRED  X__b7,r4,band_HIGH_PAIRED 
 X__b7,r9,band_LOW_PAIRED  X__b7,r10,band_HIGH_PAIRED 
 X__b1,r6,band_LOW_PAIRED  X__b1,r0,band_LOW_PAIRED  X__b8,r4,band_UNPAIRED 
 X__b2,r5,band_UNPAIRED  X__b1,r10,band_UNPAIRED  X__b8,r5,band_UNPAIRED 
 X__b2,r6,band_UNPAIRED  X__b8,r6,band_UNPAIRED  X__b1,r11,band_UNPAIRED 
 X__b2,r8,band_UNPAIRED  X__b2,r7,band_UNPAIRED  X__b4,r11,band_LOW_PAIRED 
 X__b9,r7,band_LOW_PAIRED  X__b8,r13,band_UNPAIRED  X__b3,r2,band_HIGH_PAIRED 
 X__b9,r12,band_UNPAIRED  X__b8,r3,band_LOW_PAIRED  X__b4,r13,band_UNPAIRED 
 X__b6,r0,band_HIGH_PAIRED  X__b9,r4,band_UNPAIRED  X__b5,r9,band_LOW_PAIRED 
 X__b8,r9,band_LOW_PAIRED  X__b5,r13,band_LOW_PAIRED  X__b3,r4,band_LOW_PAIRED 
 X__b2,r0,band_UNPAIRED  X__b1,r7,band_UNPAIRED  X__b4,r11,band_HIGH_PAIRED 
 X__b9,r1,band_LOW_PAIRED  X__b5,r11,band_UNPAIRED  X__b2,r7,band_HIGH_PAIRED 
 X__b0,r10,band_LOW_PAIRED  X__b7,r5,band_UNPAIRED  X__b9,r2,band_UNPAIRED 
 X__b3,r6,band_UNPAIRED  X__b1,r9,band_UNPAIRED  X__b4,r9,band_LOW_PAIRED 
 X__b2,r0,band_LOW_PAIRED  X__b2,r6,band_LOW_PAIRED 
 X__b3,r11,band_HIGH_PAIRED  X__b9,r0,band_HIGH_PAIRED  X__b3,r4,band_UNPAIRED 
 X__b8,r11,band_LOW_PAIRED  X__b0,r3,band_UNPAIRED  X__b6,r1,band_HIGH_PAIRED 
 X__b1,r12,band_LOW_PAIRED  X__b0,r1,band_UNPAIRED  X__b7,r7,band_UNPAIRED 
 X__b1,r4,band_LOW_PAIRED  X__b3,r12,band_HIGH_PAIRED 
 X__b3,r0,band_LOW_PAIRED  X__b9,r3,band_LOW_PAIRED  X__b7,r6,band_UNPAIRED 
 X__b7,r7,band_LOW_PAIRED  X__b8,r13,band_LOW_PAIRED 
 X__b7,r13,band_LOW_PAIRED  X__b5,r5,band_HIGH_PAIRED 
 X__b8,r11,band_HIGH_PAIRED  X__b9,r4,band_HIGH_PAIRED 
 X__b0,r8,band_LOW_PAIRED  X__b4,r2,band_LOW_PAIRED  X__b8,r0,band_HIGH_PAIRED 
 X__b2,r13,band_UNPAIRED  X__b1,r7,band_HIGH_PAIRED  X__b8,r9,band_HIGH_PAIRED 
 X__b3,r1,band_HIGH_PAIRED  X__b5,r0,band_LOW_PAIRED 
 X__b7,r5,band_HIGH_PAIRED  X__b6,r5,band_HIGH_PAIRED 
 X__b2,r4,band_LOW_PAIRED  X__b4,r3,band_UNPAIRED  X__b6,r8,band_UNPAIRED 
 X__b0,r13,band_UNPAIRED  X__b6,r11,band_HIGH_PAIRED  X__b3,r5,band_UNPAIRED 
 X__b1,r8,band_UNPAIRED  X__b7,r11,band_HIGH_PAIRED  X__b8,r7,band_LOW_PAIRED 
 X__b4,r0,band_LOW_PAIRED  X__b9,r9,band_HIGH_PAIRED 
 X__b4,r1,band_HIGH_PAIRED  X__b9,r3,band_UNPAIRED  X__b5,r10,band_UNPAIRED 
 X__b7,r0,band_HIGH_PAIRED  X__b0,r7,band_HIGH_PAIRED 
 X__b3,r2,band_LOW_PAIRED  X__b9,r5,band_LOW_PAIRED  X__b0,r2,band_UNPAIRED 
 X__b6,r0,band_UNPAIRED  X__b1,r12,band_HIGH_PAIRED  X__b2,r8,band_LOW_PAIRED 
 X__b5,r5,band_UNPAIRED  X__b6,r4,band_HIGH_PAIRED  X__b4,r1,band_LOW_PAIRED 
 X__b3,r10,band_LOW_PAIRED  X__b3,r6,band_HIGH_PAIRED 
 X__b6,r6,band_LOW_PAIRED  X__b0,r3,band_LOW_PAIRED  X__b9,r2,band_HIGH_PAIRED 
 X__b6,r9,band_UNPAIRED  X__b3,r11,band_UNPAIRED  X__b2,r3,band_HIGH_PAIRED 
 X__b5,r7,band_LOW_PAIRED  X__b4,r1,band_UNPAIRED  X__b7,r8,band_HIGH_PAIRED 
 X__b1,r10,band_HIGH_PAIRED  X__b0,r8,band_HIGH_PAIRED 
 X__b2,r2,band_HIGH_PAIRED  X__b5,r3,band_UNPAIRED  X__b5,r11,band_LOW_PAIRED 
 X__b9,r11,band_HIGH_PAIRED  X__b7,r10,band_UNPAIRED  X__b7,r0,band_LOW_PAIRED 
 X__b1,r9,band_LOW_PAIRED  X__b0,r12,band_LOW_PAIRED 
 X__b6,r12,band_HIGH_PAIRED  X__b5,r9,band_UNPAIRED  X__b5,r1,band_HIGH_PAIRED 
 X__b5,r7,band_UNPAIRED  X__b9,r1,band_HIGH_PAIRED  X__b4,r9,band_HIGH_PAIRED 
 X__b6,r5,band_UNPAIRED  X__b6,r13,band_HIGH_PAIRED  X__b6,r13,band_UNPAIRED 
 X__b5,r1,band_UNPAIRED  X__b2,r1,band_LOW_PAIRED  X__b3,r13,band_UNPAIRED 
 X__b6,r1,band_LOW_PAIRED  X__b0,r9,band_HIGH_PAIRED  X__b6,r7,band_UNPAIRED 
 X__b1,r0,band_HIGH_PAIRED  X__b6,r13,band_LOW_PAIRED 
 X__b9,r13,band_LOW_PAIRED  X__b6,r3,band_UNPAIRED  X__b8,r4,band_LOW_PAIRED 
 X__b6,r11,band_UNPAIRED  X__b7,r9,band_UNPAIRED  X__b2,r13,band_LOW_PAIRED 
 X__b5,r0,band_HIGH_PAIRED  X__b9,r10,band_HIGH_PAIRED 
 X__b2,r1,band_HIGH_PAIRED  X__b1,r2,band_LOW_PAIRED  X__b7,r5,band_LOW_PAIRED 
 X__b5,r2,band_LOW_PAIRED  X__b7,r9,band_HIGH_PAIRED 
 X__b5,r2,band_HIGH_PAIRED  X__b6,r1,band_UNPAIRED  X__b7,r6,band_HIGH_PAIRED 
 X__b3,r3,band_UNPAIRED  X__b0,r4,band_UNPAIRED  X__b2,r11,band_LOW_PAIRED 
 X__b4,r9,band_UNPAIRED  GlobalBidder_W_i_9  GlobalBidder_W_i_8 
 GlobalBidder_W_i_7  X__b3,r9,band_LOW_PAIRED  X__b9,r0,band_LOW_PAIRED 
 X__b0,r11,band_UNPAIRED  X__b1,r11,band_LOW_PAIRED 
 X__b9,r12,band_HIGH_PAIRED  X__b6,r2,band_HIGH_PAIRED 
 X__b6,r6,band_HIGH_PAIRED  X__b1,r1,band_HIGH_PAIRED  X__b2,r12,band_UNPAIRED 
 X__b0,r13,band_HIGH_PAIRED  X__b4,r8,band_LOW_PAIRED 
 X__b9,r11,band_LOW_PAIRED  X__b4,r12,band_LOW_PAIRED 
 X__b4,r10,band_HIGH_PAIRED  X__b3,r7,band_HIGH_PAIRED 
 X__b2,r5,band_HIGH_PAIRED  X__b9,r13,band_UNPAIRED  X__b2,r1,band_UNPAIRED 
 X__b1,r11,band_HIGH_PAIRED  X__b3,r7,band_UNPAIRED  X__b0,r8,band_UNPAIRED 
 X__b4,r5,band_UNPAIRED  X__b1,r13,band_HIGH_PAIRED  X__b3,r7,band_LOW_PAIRED 
 X__b6,r4,band_LOW_PAIRED  X__b7,r12,band_UNPAIRED  X__b1,r6,band_UNPAIRED 
 X__b0,r6,band_HIGH_PAIRED  X__b3,r13,band_HIGH_PAIRED 
 X__b8,r1,band_LOW_PAIRED  X__b4,r7,band_HIGH_PAIRED 
 X__b3,r5,band_HIGH_PAIRED  X__b0,r1,band_LOW_PAIRED 
 X__b6,r10,band_HIGH_PAIRED  X__b8,r1,band_HIGH_PAIRED 
 X__b4,r6,band_LOW_PAIRED  X__b4,r12,band_UNPAIRED  X__b5,r12,band_UNPAIRED 
 X__b0,r0,band_UNPAIRED  X__b7,r2,band_LOW_PAIRED  X__b5,r5,band_LOW_PAIRED 
 X__b3,r12,band_LOW_PAIRED  X__b9,r3,band_HIGH_PAIRED 
 X__b2,r5,band_LOW_PAIRED  X__b8,r8,band_HIGH_PAIRED 
 X__b3,r3,band_HIGH_PAIRED  X__b3,r13,band_LOW_PAIRED 
 X__b0,r5,band_HIGH_PAIRED  X__b0,r0,band_LOW_PAIRED  X__b6,r3,band_LOW_PAIRED 
 X__b9,r8,band_UNPAIRED  X__b4,r10,band_LOW_PAIRED  X__b2,r9,band_UNPAIRED 
 X__b1,r12,band_UNPAIRED  X__b1,r3,band_UNPAIRED  X__b2,r4,band_UNPAIRED 
 X__b5,r4,band_HIGH_PAIRED  X__b8,r0,band_UNPAIRED  X__b8,r1,band_UNPAIRED 
 X__b2,r3,band_UNPAIRED  X__b8,r12,band_LOW_PAIRED  X__b8,r2,band_UNPAIRED 
 X__b8,r8,band_LOW_PAIRED  X__b8,r2,band_LOW_PAIRED  X__b8,r7,band_UNPAIRED 
 X__b1,r13,band_UNPAIRED  X__b2,r0,band_HIGH_PAIRED  X__b3,r8,band_UNPAIRED 
 X__b5,r13,band_HIGH_PAIRED  X__b2,r2,band_UNPAIRED  X__b8,r9,band_UNPAIRED 
 X__b1,r13,band_LOW_PAIRED  X__b5,r3,band_LOW_PAIRED  X__b9,r6,band_UNPAIRED 
 X__b7,r1,band_UNPAIRED  X__b6,r8,band_HIGH_PAIRED  X__b1,r5,band_UNPAIRED 
 X__b7,r3,band_UNPAIRED  X__b3,r10,band_HIGH_PAIRED  X__b4,r11,band_UNPAIRED 
 X__b9,r2,band_LOW_PAIRED  X__b2,r10,band_HIGH_PAIRED 
 X__b3,r9,band_HIGH_PAIRED  X__b6,r7,band_HIGH_PAIRED 
 X__b4,r3,band_LOW_PAIRED  X__b0,r4,band_HIGH_PAIRED 
 X__b7,r3,band_HIGH_PAIRED  X__b3,r5,band_LOW_PAIRED  X__b9,r8,band_LOW_PAIRED 
 X__b9,r6,band_HIGH_PAIRED  X__b9,r0,band_UNPAIRED  X__b5,r13,band_UNPAIRED 
 X__b1,r7,band_LOW_PAIRED  X__b5,r12,band_LOW_PAIRED  X__b2,r10,band_UNPAIRED 
 X__b0,r5,band_UNPAIRED  X__b0,r10,band_UNPAIRED  X__b4,r8,band_UNPAIRED 
 X__b3,r3,band_LOW_PAIRED  X__b9,r6,band_LOW_PAIRED  X__b3,r8,band_HIGH_PAIRED 
 X__b9,r7,band_HIGH_PAIRED  X__b6,r10,band_UNPAIRED  X__b7,r10,band_LOW_PAIRED 
 X__b0,r0,band_HIGH_PAIRED  X__b4,r8,band_HIGH_PAIRED 
 X__b6,r8,band_LOW_PAIRED  X__b6,r12,band_LOW_PAIRED 
 X__b0,r10,band_HIGH_PAIRED  X__b1,r4,band_HIGH_PAIRED 
 X__b6,r3,band_HIGH_PAIRED  X__b0,r5,band_LOW_PAIRED  X__b8,r11,band_UNPAIRED 
 X__b3,r4,band_HIGH_PAIRED  X__b8,r10,band_UNPAIRED  X__b7,r12,band_LOW_PAIRED 
 X__b7,r2,band_UNPAIRED  X__b8,r13,band_HIGH_PAIRED  X__b9,r4,band_LOW_PAIRED 
 X__b8,r8,band_UNPAIRED  X__b8,r6,band_LOW_PAIRED  X__b7,r7,band_HIGH_PAIRED 
 X__b8,r7,band_HIGH_PAIRED  X__b1,r4,band_UNPAIRED  X__b3,r9,band_UNPAIRED 
 X__b5,r10,band_LOW_PAIRED  X__b3,r1,band_LOW_PAIRED 
 X__b8,r2,band_HIGH_PAIRED  X__b0,r7,band_LOW_PAIRED 
 X__b5,r8,band_HIGH_PAIRED  X__b6,r10,band_LOW_PAIRED 
 X__b0,r13,band_LOW_PAIRED  X__b9,r7,band_UNPAIRED  X__b6,r4,band_UNPAIRED 
 X__b1,r9,band_HIGH_PAIRED  X__b4,r7,band_UNPAIRED  X__b2,r3,band_LOW_PAIRED 
 X__b0,r6,band_UNPAIRED  X__b4,r10,band_UNPAIRED  X__b5,r3,band_HIGH_PAIRED 
 X__b1,r5,band_LOW_PAIRED  X__b9,r13,band_HIGH_PAIRED 
 X__b7,r8,band_LOW_PAIRED  X__b2,r4,band_HIGH_PAIRED  X__b3,r1,band_UNPAIRED 
End
