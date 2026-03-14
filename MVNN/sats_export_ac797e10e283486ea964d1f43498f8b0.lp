\ENCODING=ISO-8859-1
\Problem name: ilog.cplex

Maximize
 obj1: v__9 + v__8 + v__7 + v__6 + v__5 + v__4 + v__3 + v__2 + v__1 + v__0
Subject To
 c1:    - aux_cap_helper__b0,r13,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b0,r13,band_HIGH_PAIRED__Z3 <= 0
 c2:    aux_cap_helper__b4,r6,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b4,r6,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b4,r6,band_HIGH_PAIRED__Z3  = 1
 c3:    - 2 aux_cap_helper__b1,r7,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b1,r7,band_HIGH_PAIRED__condX2 <= 0
 c4:    0.158453086904992 sv_function_helpervar_b7,r10__Z2
        - sv_function_helpervar_b7,r10__condX2 <= 0
 c5:    - 2 aux_cap_helper__b6,r3,band_UNPAIRED__Z2
        + aux_cap_helper__b6,r3,band_UNPAIRED__condX2 <= 0
 c6:    - aux_cap_helper__b1,r4,band_UNPAIRED__Z1
        + aux_cap_helper__b1,r4,band_UNPAIRED__condX1 <= 0
 c7:    - X__b8,r4,band_HIGH_PAIRED
        + aux_cap_helper__b8,r4,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b8,r4,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b8,r4,band_HIGH_PAIRED__condX3  = 0
 c8:    - 2 aux_cap_helper__b1,r4,band_UNPAIRED__Z2
        + aux_cap_helper__b1,r4,band_UNPAIRED__condX2 <= 0
 c9:    - aux_cap_helper__b8,r11,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b8,r11,band_HIGH_PAIRED__condX1 <= 0
 c10:   aux_cap_helper__b8,r0,band_UNPAIRED__Z2
        - aux_cap_helper__b8,r0,band_UNPAIRED__condX2 <= 0
 c11:   - aux_cap_b2,r6,band_HIGH_PAIRED
        + 1.8253501688 aux_cap_helper__b2,r6,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b2,r6,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b2,r6,band_HIGH_PAIRED__condX1
        + 1.21690011250031 aux_cap_helper__b2,r6,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b2,r6,band_HIGH_PAIRED__Z2  = 0
 c12:   aux_cap_helper__b4,r1,band_LOW_PAIRED__condX1
        - aux_cap_helper__b4,r1,band_LOW_PAIRED__Z1 <= 0
 c13:   - aux_cap_helper__b3,r0,band_HIGH_PAIRED__condX1 <= 0
 c14:   aux_cap_helper__b2,r0,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b2,r0,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b2,r0,band_HIGH_PAIRED__Z1  = 1
 c15:   5.43414635547902 sv_function_helpervar_b6,r4__Z3
        - sv_function_helpervar_b6,r4__condX3 <= 0
 c16:   - aux_cap_helper__b5,r0,band_LOW_PAIRED__condX1 <= 0
 c17:   aux_cap_helper__b8,r12,band_UNPAIRED__Z1
        + aux_cap_helper__b8,r12,band_UNPAIRED__Z2  = 1
 c18:   - 3 aux_cap_helper__b1,r1,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b1,r1,band_HIGH_PAIRED__condX3 <= 0
 c19:   - 0.8112667416 aux_cap_helper__b4,r7,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b4,r7,band_HIGH_PAIRED__Z3
        + 1.8253501688 aux_cap_helper__b4,r7,band_HIGH_PAIRED__condX3
        + 2.0281668543 aux_cap_helper__b4,r7,band_HIGH_PAIRED__condX1
        + 2.8394335959 aux_cap_helper__b4,r7,band_HIGH_PAIRED__condX2
        - aux_cap_b4,r7,band_HIGH_PAIRED  = 0
 c20:   sv_function_helpervar_b1,r11__condX1
        - 0.1584530868979 sv_function_helpervar_b1,r11__Z1 <= 0
 c21:   aux_cap_b8,r4,band_HIGH_PAIRED - aux_c_b8,r4
        + aux_cap_b8,r4,band_LOW_PAIRED + aux_cap_b8,r4,band_UNPAIRED  = 0
 c22:   - 3 aux_cap_helper__b8,r9,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b8,r9,band_HIGH_PAIRED__condX3 <= 0
 c23:   - aux_cap_helper__b8,r3,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b8,r3,band_HIGH_PAIRED__Z2 <= 0
 c24:   - aux_cap_helper__b0,r4,band_HIGH_PAIRED__condX1 <= 0
 c25:   aux_cap_helper__b3,r1,band_UNPAIRED__Z2
        - aux_cap_helper__b3,r1,band_UNPAIRED__condX2 <= 0
 c26:   - aux_cap_helper__b4,r6,band_UNPAIRED__condX1 <= 0
 c27:   - 2 aux_cap_helper__b8,r11,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b8,r11,band_HIGH_PAIRED__condX2 <= 0
 c28:   aux_cap_helper__b4,r3,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b4,r3,band_HIGH_PAIRED__Z1 <= 0
 c29:   - 20668026.8748422 GlobalBidder_hat{W}__i8,k0 - GlobalBidder_PSI__i8,k0
        + aux_Omega_b8,r12 + aux_Omega_b8,r13 + aux_Omega_b8,r10
        + aux_Omega_b8,r11 + aux_Omega_b8,r5 + aux_Omega_b8,r6
        + aux_Omega_b8,r7 + aux_Omega_b8,r8 + aux_Omega_b8,r9 + aux_Omega_b8,r0
        + aux_Omega_b8,r1 + aux_Omega_b8,r2 + aux_Omega_b8,r3 + aux_Omega_b8,r4
        >= -20668026.8748422
 c30:   aux_cap_helper__b9,r10,band_UNPAIRED__Z1
        + aux_cap_helper__b9,r10,band_UNPAIRED__Z2  = 1
 c31:   0.888446608 aux_cap_helper__b9,r13,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b9,r13,band_UNPAIRED__condX1
        - 0.253841888 aux_cap_helper__b9,r13,band_UNPAIRED__Z2
        - aux_cap_b9,r13,band_UNPAIRED  = 0
 c32:   4.4504289328 aux_cap_helper__b3,r4,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b3,r4,band_LOW_PAIRED__condX1
        - aux_cap_b3,r4,band_LOW_PAIRED
        - 1.2715511236 aux_cap_helper__b3,r4,band_LOW_PAIRED__Z2  = 0
 c33:   aux_cap_helper__b0,r8,band_LOW_PAIRED__condX2
        + aux_cap_helper__b0,r8,band_LOW_PAIRED__condX1
        - X__b0,r8,band_LOW_PAIRED  = 0
 c34:   - aux_cap_helper__b1,r7,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b1,r7,band_HIGH_PAIRED__condX1 <= 0
 c35:   aux_cap_helper__b5,r10,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b5,r10,band_HIGH_PAIRED__condX2 <= 0
 c36:   18.0804758813 sv_function_helpervar_b3,r3__condX3
        + 55.6489151131 sv_function_helpervar_b3,r3__condX2
        + 1152.1877365853 sv_function_helpervar_b3,r3__condX1 - aux_sv_b3,r3
        + 173.749961175029 sv_function_helpervar_b3,r3__Z2
        + 389.685958265197 sv_function_helpervar_b3,r3__Z3  = 0
 c37:   - sv_function_helpervar_b6,r0__condX1 <= 0
 c38:   - sv_function_helpervar_b3,r10__condX3
        + 5.86020404779476 sv_function_helpervar_b3,r10__Z3 <= 0
 c39:   148.226284890282 sv_function_helpervar_b5,r6__Z2
        + 318.569857473697 sv_function_helpervar_b5,r6__Z3 - aux_sv_b5,r6
        + 978.901873111 sv_function_helpervar_b5,r6__condX1
        + 43.4433863348 sv_function_helpervar_b5,r6__condX2
        + 16.1506295835 sv_function_helpervar_b5,r6__condX3  = 0
 c40:   - aux_Omega_b7,r2 + 466.876153926107 aux_sv_b7,r2  = 0
 c41:   aux_cap_helper__b8,r9,band_LOW_PAIRED__condX1
        + aux_cap_helper__b8,r9,band_LOW_PAIRED__condX2
        - X__b8,r9,band_LOW_PAIRED  = 0
 c42:   aux_cap_helper__b9,r7,band_LOW_PAIRED__Z1
        + aux_cap_helper__b9,r7,band_LOW_PAIRED__Z2  = 1
 c43:   aux_cap_helper__b0,r2,band_UNPAIRED__Z2
        + aux_cap_helper__b0,r2,band_UNPAIRED__Z1  = 1
 c44:   2 aux_cap_helper__b5,r10,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b5,r10,band_HIGH_PAIRED__condX3 <= 0
 c45:   - aux_cap_helper__b1,r4,band_LOW_PAIRED__condX1 <= 0
 c46:   aux_cap_b5,r11,band_UNPAIRED + aux_cap_b5,r11,band_LOW_PAIRED
        + aux_cap_b5,r11,band_HIGH_PAIRED - aux_c_b5,r11  = 0
 c47:   - 0.8112667416 aux_cap_helper__b6,r8,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b6,r8,band_HIGH_PAIRED__Z3
        + 1.8253501688 aux_cap_helper__b6,r8,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b6,r8,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b6,r8,band_HIGH_PAIRED__condX1
        - aux_cap_b6,r8,band_HIGH_PAIRED  = 0
 c48:   - aux_cap_helper__b6,r6,band_LOW_PAIRED__condX1 <= 0
 c49:   sv_function_helpervar_b1,r12__Z3 + sv_function_helpervar_b1,r12__Z2
        + sv_function_helpervar_b1,r12__Z1  = 1
 c50:   - sv_function_helpervar_b5,r2__condX1 <= 0
 c51:   - 2 aux_cap_helper__b7,r0,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b7,r0,band_HIGH_PAIRED__condX2 <= 0
 c52:   - aux_cap_helper__b4,r8,band_LOW_PAIRED__condX2
        + aux_cap_helper__b4,r8,band_LOW_PAIRED__Z2 <= 0
 c53:   - aux_cap_helper__b9,r5,band_UNPAIRED__condX1 <= 0
 c54:   - aux_cap_helper__b9,r11,band_LOW_PAIRED__condX1 <= 0
 c55:   - 2 aux_cap_helper__b0,r5,band_LOW_PAIRED__Z2
        + aux_cap_helper__b0,r5,band_LOW_PAIRED__condX2 <= 0
 c56:   aux_cap_helper__b6,r4,band_LOW_PAIRED__Z1
        + aux_cap_helper__b6,r4,band_LOW_PAIRED__Z2  = 1
 c57:   1.21690011250031 aux_cap_helper__b8,r7,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b8,r7,band_HIGH_PAIRED__Z2
        - aux_cap_b8,r7,band_HIGH_PAIRED
        + 1.8253501688 aux_cap_helper__b8,r7,band_HIGH_PAIRED__condX3
        + 2.0281668543 aux_cap_helper__b8,r7,band_HIGH_PAIRED__condX1
        + 2.8394335959 aux_cap_helper__b8,r7,band_HIGH_PAIRED__condX2  = 0
 c58:   sv_function_helpervar_b8,r8__condX3
        - 15.8453086887446 sv_function_helpervar_b8,r8__Z3 <= 0
 c59:   aux_cap_helper__b6,r4,band_UNPAIRED__Z2
        + aux_cap_helper__b6,r4,band_UNPAIRED__Z1  = 1
 c60:   0.158453086901135 sv_function_helpervar_b2,r6__Z2
        - sv_function_helpervar_b2,r6__condX2 <= 0
 c61:   aux_cap_helper__b3,r2,band_LOW_PAIRED__Z2
        - aux_cap_helper__b3,r2,band_LOW_PAIRED__condX2 <= 0
 c62:   - 0.15845308689553 sv_function_helpervar_b6,r4__Z1
        + sv_function_helpervar_b6,r4__condX1 <= 0
 c63:   sv_function_helpervar_b1,r7__Z2 + sv_function_helpervar_b1,r7__Z1
        + sv_function_helpervar_b1,r7__Z3  = 1
 c64:   - aux_Omega_b3,r12 + 948.397037648752 aux_sv_b3,r12  = 0
 c65:   - aux_cap_helper__b2,r11,band_UNPAIRED__condX2
        + aux_cap_helper__b2,r11,band_UNPAIRED__Z2 <= 0
 c66:   sv_function_helpervar_b7,r4__Z1 + sv_function_helpervar_b7,r4__Z2
        + sv_function_helpervar_b7,r4__Z3  = 1
 c67:   1839.54961954162 aux_sv_b1,r11 - aux_Omega_b1,r11  = 0
 c68:   aux_cap_helper__b1,r13,band_LOW_PAIRED__Z2
        + aux_cap_helper__b1,r13,band_LOW_PAIRED__Z1  = 1
 c69:   - aux_sv_b9,r11 + 1995.2925261198 sv_function_helpervar_b9,r11__condX1
        + 39.4178990341 sv_function_helpervar_b9,r11__condX3
        + 70.262779159 sv_function_helpervar_b9,r11__condX2
        + 305.026905934037 sv_function_helpervar_b9,r11__Z2
        + 546.375148579275 sv_function_helpervar_b9,r11__Z3  = 0
 c70:   2.0281668543 aux_cap_helper__b0,r6,band_HIGH_PAIRED__condX1
        + 1.8253501688 aux_cap_helper__b0,r6,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b0,r6,band_HIGH_PAIRED__condX2
        - aux_cap_b0,r6,band_HIGH_PAIRED
        + 1.21690011250031 aux_cap_helper__b0,r6,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b0,r6,band_HIGH_PAIRED__Z2  = 0
 c71:   - aux_Omega_b4,r1 + 295.473282230016 aux_sv_b4,r1  = 0
 c72:   aux_cap_helper__b3,r1,band_LOW_PAIRED__Z1
        + aux_cap_helper__b3,r1,band_LOW_PAIRED__Z2  = 1
 c73:   aux_cap_helper__b3,r13,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b3,r13,band_HIGH_PAIRED__Z1 <= 0
 c74:   - sv_function_helpervar_b7,r11__condX1 <= 0
 c75:   - aux_Omega_b3,r7 + 2152.51974086741 aux_sv_b3,r7  = 0
 c76:   - aux_cap_b6,r6,band_HIGH_PAIRED
        + 1.8253501688 aux_cap_helper__b6,r6,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b6,r6,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b6,r6,band_HIGH_PAIRED__condX1
        + 1.21690011250031 aux_cap_helper__b6,r6,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b6,r6,band_HIGH_PAIRED__Z2  = 0
 c77:   - aux_cap_helper__b6,r3,band_UNPAIRED__Z1
        + aux_cap_helper__b6,r3,band_UNPAIRED__condX1 <= 0
 c78:   aux_cap_helper__b4,r3,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b4,r3,band_HIGH_PAIRED__Z2 <= 0
 c79:   sv_function_helpervar_b9,r11__condX2
        - 7.82458034098202 sv_function_helpervar_b9,r11__Z2 <= 0
 c80:   2 aux_cap_helper__b5,r7,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b5,r7,band_HIGH_PAIRED__condX3 <= 0
 c81:   2.0281668543 aux_cap_helper__b2,r7,band_HIGH_PAIRED__condX1
        + 2.8394335959 aux_cap_helper__b2,r7,band_HIGH_PAIRED__condX2
        + 1.8253501688 aux_cap_helper__b2,r7,band_HIGH_PAIRED__condX3
        - aux_cap_b2,r7,band_HIGH_PAIRED
        - 0.8112667416 aux_cap_helper__b2,r7,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b2,r7,band_HIGH_PAIRED__Z3  = 0
 c82:   - aux_cap_helper__b0,r3,band_HIGH_PAIRED__condX1 <= 0
 c83:   - sv_function_helpervar_b7,r9__condX1 <= 0
 c84:   - aux_cap_helper__b9,r6,band_LOW_PAIRED__Z1
        + aux_cap_helper__b9,r6,band_LOW_PAIRED__condX1 <= 0
 c85:   - aux_cap_helper__b9,r11,band_UNPAIRED__Z1
        + aux_cap_helper__b9,r11,band_UNPAIRED__condX1 <= 0
 c86:   - aux_cap_helper__b4,r1,band_UNPAIRED__condX2
        + aux_cap_helper__b4,r1,band_UNPAIRED__Z2 <= 0
 c87:   - aux_cap_helper__b3,r6,band_UNPAIRED__condX1 <= 0
 c88:   sv_function_helpervar_b6,r1__condX3
        - 15.8453086887446 sv_function_helpervar_b6,r1__Z3 <= 0
 c89:   sv_function_helpervar_b3,r10__condX3
        + sv_function_helpervar_b3,r10__condX1
        + sv_function_helpervar_b3,r10__condX2 - aux_c_b3,r10  = 0
 c90:   2.0281668543 aux_cap_helper__b0,r4,band_HIGH_PAIRED__condX1
        + 1.8253501688 aux_cap_helper__b0,r4,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b0,r4,band_HIGH_PAIRED__condX2
        - aux_cap_b0,r4,band_HIGH_PAIRED
        + 1.21690011250031 aux_cap_helper__b0,r4,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b0,r4,band_HIGH_PAIRED__Z2  = 0
 c91:   sv_function_helpervar_b7,r12__condX3
        - 15.8453086887446 sv_function_helpervar_b7,r12__Z3 <= 0
 c92:   1.21690011250031 aux_cap_helper__b8,r9,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b8,r9,band_HIGH_PAIRED__Z2
        - aux_cap_b8,r9,band_HIGH_PAIRED
        + 1.8253501688 aux_cap_helper__b8,r9,band_HIGH_PAIRED__condX3
        + 2.0281668543 aux_cap_helper__b8,r9,band_HIGH_PAIRED__condX1
        + 2.8394335959 aux_cap_helper__b8,r9,band_HIGH_PAIRED__condX2  = 0
 c93:   aux_cap_helper__b5,r3,band_UNPAIRED__condX1
        - aux_cap_helper__b5,r3,band_UNPAIRED__Z1 <= 0
 c94:   - 2 aux_cap_helper__b9,r11,band_UNPAIRED__Z2
        + aux_cap_helper__b9,r11,band_UNPAIRED__condX2 <= 0
 c95:   aux_cap_helper__b0,r12,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b0,r12,band_HIGH_PAIRED__condX2 <= 0
 c96:   2 aux_cap_helper__b8,r2,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b8,r2,band_HIGH_PAIRED__condX3 <= 0
 c97:   - sv_function_helpervar_b2,r8__condX1 <= 0
 c98:   aux_cap_helper__b5,r11,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b5,r11,band_HIGH_PAIRED__condX2 <= 0
 c99:   aux_cap_helper__b9,r3,band_LOW_PAIRED__condX1
        - aux_cap_helper__b9,r3,band_LOW_PAIRED__Z1 <= 0
 c100:  - aux_cap_helper__b5,r13,band_UNPAIRED__condX2
        + aux_cap_helper__b5,r13,band_UNPAIRED__Z2 <= 0
 c101:  - X__b8,r10,band_HIGH_PAIRED
        + aux_cap_helper__b8,r10,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b8,r10,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b8,r10,band_HIGH_PAIRED__condX1  = 0
 c102:  - 2 aux_cap_helper__b1,r8,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b1,r8,band_HIGH_PAIRED__condX2 <= 0
 c103:  aux_cap_helper__b3,r13,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b3,r13,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b3,r13,band_HIGH_PAIRED__Z3  = 1
 c104:  aux_cap_helper__b5,r3,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b5,r3,band_UNPAIRED__Z2 <= 0
 c105:  2 aux_cap_helper__b0,r12,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b0,r12,band_HIGH_PAIRED__condX3 <= 0
 c106:  2 aux_cap_helper__b5,r11,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b5,r11,band_HIGH_PAIRED__condX3 <= 0
 c107:  aux_cap_helper__b2,r4,band_UNPAIRED__condX1
        - aux_cap_helper__b2,r4,band_UNPAIRED__Z1 <= 0
 c108:  aux_cap_b3,r4,band_LOW_PAIRED + aux_cap_b3,r4,band_UNPAIRED
        - aux_c_b3,r4 + aux_cap_b3,r4,band_HIGH_PAIRED  = 0
 c109:  aux_cap_helper__b5,r7,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b5,r7,band_HIGH_PAIRED__condX2 <= 0
 c110:  aux_cap_helper__b0,r1,band_LOW_PAIRED__condX2
        + aux_cap_helper__b0,r1,band_LOW_PAIRED__condX1
        - X__b0,r1,band_LOW_PAIRED  = 0
 c111:  aux_cap_helper__b8,r4,band_LOW_PAIRED__Z2
        - aux_cap_helper__b8,r4,band_LOW_PAIRED__condX2 <= 0
 c112:  aux_cap_helper__b9,r3,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b9,r3,band_LOW_PAIRED__Z2 <= 0
 c113:  X__b5,r2,band_UNPAIRED + X__b4,r2,band_UNPAIRED
        + X__b3,r2,band_UNPAIRED + X__b6,r2,band_UNPAIRED
        + X__b1,r2,band_UNPAIRED + X__b9,r2,band_UNPAIRED
        + X__b0,r2,band_UNPAIRED + X__b8,r2,band_UNPAIRED
        + X__b2,r2,band_UNPAIRED + X__b7,r2,band_UNPAIRED <= 2
 c114:  sv_function_helpervar_b2,r3__Z1 + sv_function_helpervar_b2,r3__Z2
        + sv_function_helpervar_b2,r3__Z3  = 1
 c115:  - aux_cap_b6,r9,band_HIGH_PAIRED
        - 0.8112667416 aux_cap_helper__b6,r9,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b6,r9,band_HIGH_PAIRED__Z3
        + 1.8253501688 aux_cap_helper__b6,r9,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b6,r9,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b6,r9,band_HIGH_PAIRED__condX1  = 0
 c116:  0.0102040816326531 X__b9,r1,band_UNPAIRED
        + 0.0102040816326531 X__b9,r1,band_LOW_PAIRED
        + 0.0102040816326531 X__b9,r1,band_HIGH_PAIRED
        - GlobalBidder_W_ir__b9,r1 <= 0
 c117:  - 0.253841888 aux_cap_helper__b8,r9,band_UNPAIRED__Z2
        - aux_cap_b8,r9,band_UNPAIRED
        + 0.888446608 aux_cap_helper__b8,r9,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b8,r9,band_UNPAIRED__condX1  = 0
 c118:  - aux_cap_b2,r5,band_UNPAIRED
        - 0.253841888 aux_cap_helper__b2,r5,band_UNPAIRED__Z2
        + 0.888446608 aux_cap_helper__b2,r5,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b2,r5,band_UNPAIRED__condX1  = 0
 c119:  - aux_cap_helper__b8,r10,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b8,r10,band_HIGH_PAIRED__condX1 <= 0
 c120:  aux_cap_helper__b2,r5,band_LOW_PAIRED__Z2
        + aux_cap_helper__b2,r5,band_LOW_PAIRED__Z1  = 1
 c121:  - aux_cap_b0,r0,band_LOW_PAIRED
        - 1.2715511236 aux_cap_helper__b0,r0,band_LOW_PAIRED__Z2
        + 4.4504289328 aux_cap_helper__b0,r0,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b0,r0,band_LOW_PAIRED__condX1  = 0
 c122:  aux_cap_helper__b0,r4,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b0,r4,band_UNPAIRED__Z2 <= 0
 c123:  aux_cap_helper__b6,r3,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b6,r3,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b6,r3,band_HIGH_PAIRED__condX1
        - X__b6,r3,band_HIGH_PAIRED  = 0
 c124:  aux_cap_helper__b8,r4,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b8,r4,band_HIGH_PAIRED__condX2 <= 0
 c125:  aux_cap_helper__b1,r2,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b1,r2,band_HIGH_PAIRED__Z3 <= 0
 c126:  - 2 aux_cap_helper__b8,r10,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b8,r10,band_HIGH_PAIRED__condX2 <= 0
 c127:  2.0281668543 aux_cap_helper__b4,r6,band_HIGH_PAIRED__condX1
        + 1.8253501688 aux_cap_helper__b4,r6,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b4,r6,band_HIGH_PAIRED__condX2
        - aux_cap_b4,r6,band_HIGH_PAIRED
        - 0.8112667416 aux_cap_helper__b4,r6,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b4,r6,band_HIGH_PAIRED__Z3  = 0
 c128:  - aux_Omega_b7,r9 + 447.449113415885 aux_sv_b7,r9  = 0
 c129:  - 3 aux_cap_helper__b8,r8,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b8,r8,band_HIGH_PAIRED__condX3 <= 0
 c130:  - aux_cap_helper__b0,r5,band_LOW_PAIRED__Z1
        + aux_cap_helper__b0,r5,band_LOW_PAIRED__condX1 <= 0
 c131:  - X__b7,r0,band_LOW_PAIRED
        + aux_cap_helper__b7,r0,band_LOW_PAIRED__condX1
        + aux_cap_helper__b7,r0,band_LOW_PAIRED__condX2  = 0
 c132:  aux_cap_helper__b5,r7,band_LOW_PAIRED__condX1
        - aux_cap_helper__b5,r7,band_LOW_PAIRED__Z1 <= 0
 c133:  1.8253501688 aux_cap_helper__b0,r5,band_HIGH_PAIRED__condX3
        + 2.0281668543 aux_cap_helper__b0,r5,band_HIGH_PAIRED__condX1
        + 2.8394335959 aux_cap_helper__b0,r5,band_HIGH_PAIRED__condX2
        - aux_cap_b0,r5,band_HIGH_PAIRED
        - 0.8112667416 aux_cap_helper__b0,r5,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b0,r5,band_HIGH_PAIRED__Z3  = 0
 c134:  - aux_cap_helper__b4,r2,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b4,r2,band_HIGH_PAIRED__condX1 <= 0
 c135:  - aux_cap_helper__b1,r8,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b1,r8,band_HIGH_PAIRED__condX1 <= 0
 c136:  X__b7,r6,band_LOW_PAIRED + X__b7,r6,band_UNPAIRED
        - GlobalBidder_W_ir__b7,r6 + X__b7,r6,band_HIGH_PAIRED >= 0
 c137:  - aux_cap_b6,r5,band_HIGH_PAIRED
        - 0.8112667416 aux_cap_helper__b6,r5,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b6,r5,band_HIGH_PAIRED__Z3
        + 2.8394335959 aux_cap_helper__b6,r5,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b6,r5,band_HIGH_PAIRED__condX1
        + 1.8253501688 aux_cap_helper__b6,r5,band_HIGH_PAIRED__condX3  = 0
 c138:  sv_function_helpervar_b6,r12__condX2
        - 5.69088093223458 sv_function_helpervar_b6,r12__Z2 <= 0
 c139:  aux_cap_helper__b2,r13,band_LOW_PAIRED__Z2
        - aux_cap_helper__b2,r13,band_LOW_PAIRED__condX2 <= 0
 c140:  1.21690011250031 aux_cap_helper__b2,r8,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b2,r8,band_HIGH_PAIRED__Z2
        - aux_cap_b2,r8,band_HIGH_PAIRED
        + 2.0281668543 aux_cap_helper__b2,r8,band_HIGH_PAIRED__condX1
        + 2.8394335959 aux_cap_helper__b2,r8,band_HIGH_PAIRED__condX2
        + 1.8253501688 aux_cap_helper__b2,r8,band_HIGH_PAIRED__condX3  = 0
 c141:  - 0.15845308693176 sv_function_helpervar_b6,r0__Z1
        + sv_function_helpervar_b6,r0__condX1 <= 0
 c142:  aux_cap_helper__b5,r7,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b5,r7,band_LOW_PAIRED__Z2 <= 0
 c143:  aux_cap_helper__b3,r3,band_LOW_PAIRED__condX1
        + aux_cap_helper__b3,r3,band_LOW_PAIRED__condX2
        - X__b3,r3,band_LOW_PAIRED  = 0
 c144:  - 2 aux_cap_helper__b4,r2,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b4,r2,band_HIGH_PAIRED__condX2 <= 0
 c145:  - 5.69096845606432 sv_function_helpervar_b3,r6__Z2
        + sv_function_helpervar_b3,r6__condX2 <= 0
 c146:  aux_cap_helper__b5,r8,band_LOW_PAIRED__Z2
        + aux_cap_helper__b5,r8,band_LOW_PAIRED__Z1  = 1
 c147:  - 0.158453086892444 sv_function_helpervar_b2,r7__Z1
        + sv_function_helpervar_b2,r7__condX1 <= 0
 c148:  - X__b0,r3,band_HIGH_PAIRED
        + aux_cap_helper__b0,r3,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b0,r3,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b0,r3,band_HIGH_PAIRED__condX1  = 0
 c149:  150.9476060615 sv_function_helpervar_b1,r6__condX1
        + 2.4467896032 sv_function_helpervar_b1,r6__condX3
        + 6.8931793105 sv_function_helpervar_b1,r6__condX2 - aux_sv_b1,r6
        + 49.8154714230929 sv_function_helpervar_b1,r6__Z3
        + 22.8258686140781 sv_function_helpervar_b1,r6__Z2  = 0
 c150:  6.24134726058085 sv_function_helpervar_b5,r6__Z3
        - sv_function_helpervar_b5,r6__condX3 <= 0
 c151:  - sv_function_helpervar_b9,r7__condX3
        + 7.41502217739354 sv_function_helpervar_b9,r7__Z3 <= 0
 c152:  - aux_cap_helper__b1,r12,band_UNPAIRED__condX1 <= 0
 c153:  aux_cap_helper__b4,r1,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b4,r1,band_LOW_PAIRED__Z2 <= 0
 c154:  - 1.2715511236 aux_cap_helper__b3,r11,band_LOW_PAIRED__Z2
        - aux_cap_b3,r11,band_LOW_PAIRED
        + 4.4504289328 aux_cap_helper__b3,r11,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b3,r11,band_LOW_PAIRED__condX1  = 0
 c155:  aux_cap_helper__b6,r7,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b6,r7,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b6,r7,band_HIGH_PAIRED__Z2  = 1
 c156:  aux_cap_helper__b8,r1,band_UNPAIRED__Z1
        + aux_cap_helper__b8,r1,band_UNPAIRED__Z2  = 1
 c157:  aux_cap_helper__b7,r0,band_UNPAIRED__Z2
        - aux_cap_helper__b7,r0,band_UNPAIRED__condX2 <= 0
 c158:  aux_cap_helper__b8,r12,band_LOW_PAIRED__Z1
        + aux_cap_helper__b8,r12,band_LOW_PAIRED__Z2  = 1
 c159:  - X__b4,r5,band_UNPAIRED + aux_cap_helper__b4,r5,band_UNPAIRED__condX2
        + aux_cap_helper__b4,r5,band_UNPAIRED__condX1  = 0
 c160:  1.21690011250031 aux_cap_helper__b8,r8,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b8,r8,band_HIGH_PAIRED__Z2
        + 2.0281668543 aux_cap_helper__b8,r8,band_HIGH_PAIRED__condX1
        + 1.8253501688 aux_cap_helper__b8,r8,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b8,r8,band_HIGH_PAIRED__condX2
        - aux_cap_b8,r8,band_HIGH_PAIRED  = 0
 c161:  sv_function_helpervar_b8,r0__Z3 + sv_function_helpervar_b8,r0__Z2
        + sv_function_helpervar_b8,r0__Z1  = 1
 c162:  - aux_cap_helper__b6,r13,band_HIGH_PAIRED__condX1 <= 0
 c163:  - aux_cap_helper__b5,r12,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b5,r12,band_HIGH_PAIRED__Z3 <= 0
 c164:  aux_cap_helper__b4,r4,band_LOW_PAIRED__condX1
        - aux_cap_helper__b4,r4,band_LOW_PAIRED__Z1 <= 0
 c165:  sv_function_helpervar_b9,r5__condX1
        - 0.15845308692186 sv_function_helpervar_b9,r5__Z1 <= 0
 c166:  sv_function_helpervar_b4,r13__Z1 + sv_function_helpervar_b4,r13__Z2
        + sv_function_helpervar_b4,r13__Z3  = 1
 c167:  aux_cap_helper__b9,r5,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b9,r5,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b9,r5,band_HIGH_PAIRED__Z3  = 1
 c168:  1.21690011250031 aux_cap_helper__b4,r4,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b4,r4,band_HIGH_PAIRED__Z2
        + 2.0281668543 aux_cap_helper__b4,r4,band_HIGH_PAIRED__condX1
        + 2.8394335959 aux_cap_helper__b4,r4,band_HIGH_PAIRED__condX2
        + 1.8253501688 aux_cap_helper__b4,r4,band_HIGH_PAIRED__condX3
        - aux_cap_b4,r4,band_HIGH_PAIRED  = 0
 c169:  - sv_function_helpervar_b8,r10__condX2
        + 0.158453086919002 sv_function_helpervar_b8,r10__Z2 <= 0
 c170:  - aux_cap_helper__b1,r8,band_LOW_PAIRED__Z1
        + aux_cap_helper__b1,r8,band_LOW_PAIRED__condX1 <= 0
 c171:  5.88835519733929 sv_function_helpervar_b6,r5__Z3
        - sv_function_helpervar_b6,r5__condX3 <= 0
 c172:  aux_cap_helper__b4,r4,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b4,r4,band_LOW_PAIRED__Z2 <= 0
 c173:  - 6.00433149309822 sv_function_helpervar_b4,r6__Z2
        + sv_function_helpervar_b4,r6__condX2 <= 0
 c174:  sv_function_helpervar_b1,r2__condX2
        - 5.66692189745523 sv_function_helpervar_b1,r2__Z2 <= 0
 c175:  - aux_cap_helper__b3,r12,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b3,r12,band_HIGH_PAIRED__condX1 <= 0
 c176:  - aux_cap_helper__b0,r9,band_LOW_PAIRED__condX2
        + aux_cap_helper__b0,r9,band_LOW_PAIRED__Z2 <= 0
 c177:  - aux_cap_helper__b7,r5,band_UNPAIRED__condX1 <= 0
 c178:  - 2 aux_cap_helper__b1,r8,band_LOW_PAIRED__Z2
        + aux_cap_helper__b1,r8,band_LOW_PAIRED__condX2 <= 0
 c179:  - 7.00044505459004 sv_function_helpervar_b7,r0__Z2
        + sv_function_helpervar_b7,r0__condX2 <= 0
 c180:  - 0.8112667416 aux_cap_helper__b8,r5,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b8,r5,band_HIGH_PAIRED__Z3
        + 2.8394335959 aux_cap_helper__b8,r5,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b8,r5,band_HIGH_PAIRED__condX1
        + 1.8253501688 aux_cap_helper__b8,r5,band_HIGH_PAIRED__condX3
        - aux_cap_b8,r5,band_HIGH_PAIRED  = 0
 c181:  - aux_Omega_b1,r5 + 627.300324018314 aux_sv_b1,r5  = 0
 c182:  aux_cap_helper__b3,r7,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b3,r7,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b3,r7,band_HIGH_PAIRED__condX3
        - X__b3,r7,band_HIGH_PAIRED  = 0
 c183:  aux_cap_b1,r7,band_UNPAIRED + aux_cap_b1,r7,band_LOW_PAIRED
        + aux_cap_b1,r7,band_HIGH_PAIRED - aux_c_b1,r7  = 0
 c184:  aux_cap_helper__b9,r0,band_UNPAIRED__Z2
        + aux_cap_helper__b9,r0,band_UNPAIRED__Z1  = 1
 c185:  - X__b2,r11,band_LOW_PAIRED
        + aux_cap_helper__b2,r11,band_LOW_PAIRED__condX2
        + aux_cap_helper__b2,r11,band_LOW_PAIRED__condX1  = 0
 c186:  0.888446608 aux_cap_helper__b4,r0,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b4,r0,band_UNPAIRED__condX1
        - aux_cap_b4,r0,band_UNPAIRED
        - 0.253841888 aux_cap_helper__b4,r0,band_UNPAIRED__Z2  = 0
 c187:  - 2 aux_cap_helper__b3,r12,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b3,r12,band_HIGH_PAIRED__condX2 <= 0
 c188:  0.63460472 aux_cap_helper__b7,r2,band_UNPAIRED__condX1
        + 0.888446608 aux_cap_helper__b7,r2,band_UNPAIRED__condX2
        - 0.253841888 aux_cap_helper__b7,r2,band_UNPAIRED__Z2
        - aux_cap_b7,r2,band_UNPAIRED  = 0
 c189:  - aux_cap_helper__b7,r13,band_LOW_PAIRED__condX1 <= 0
 c190:  X__b9,r3,band_LOW_PAIRED + X__b9,r3,band_UNPAIRED
        - GlobalBidder_W_ir__b9,r3 + X__b9,r3,band_HIGH_PAIRED >= 0
 c191:  2.8394335959 aux_cap_helper__b0,r8,band_HIGH_PAIRED__condX2
        + 1.8253501688 aux_cap_helper__b0,r8,band_HIGH_PAIRED__condX3
        + 2.0281668543 aux_cap_helper__b0,r8,band_HIGH_PAIRED__condX1
        - aux_cap_b0,r8,band_HIGH_PAIRED
        - 0.8112667416 aux_cap_helper__b0,r8,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b0,r8,band_HIGH_PAIRED__Z3  = 0
 c192:  - aux_sv_b2,r11 + 37.2452284235443 sv_function_helpervar_b2,r11__Z3
        + 17.3742116656537 sv_function_helpervar_b2,r11__Z2
        + 1.8982140391 sv_function_helpervar_b2,r11__condX3
        + 5.0677624769 sv_function_helpervar_b2,r11__condX2
        + 114.7166939165 sv_function_helpervar_b2,r11__condX1  = 0
 c193:  - aux_cap_helper__b1,r10,band_LOW_PAIRED__Z1
        + aux_cap_helper__b1,r10,band_LOW_PAIRED__condX1 <= 0
 c194:  sv_function_helpervar_b6,r3__Z1 + sv_function_helpervar_b6,r3__Z3
        + sv_function_helpervar_b6,r3__Z2  = 1
 c195:  - aux_cap_helper__b9,r2,band_UNPAIRED__Z1
        + aux_cap_helper__b9,r2,band_UNPAIRED__condX1 <= 0
 c196:  6.52419424210838 sv_function_helpervar_b2,r9__Z3
        - sv_function_helpervar_b2,r9__condX3 <= 0
 c197:  - aux_cap_b0,r3,band_HIGH_PAIRED
        + 1.8253501688 aux_cap_helper__b0,r3,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b0,r3,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b0,r3,band_HIGH_PAIRED__condX1
        + 1.21690011250031 aux_cap_helper__b0,r3,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b0,r3,band_HIGH_PAIRED__Z2  = 0
 c198:  - 15.8453086887446 sv_function_helpervar_b2,r9__Z3
        + sv_function_helpervar_b2,r9__condX3 <= 0
 c199:  - sv_function_helpervar_b5,r10__condX1 <= 0
 c200:  - 2 aux_cap_helper__b7,r1,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b7,r1,band_HIGH_PAIRED__condX2 <= 0
 c201:  - 6.01751054263764 sv_function_helpervar_b6,r8__Z2
        + sv_function_helpervar_b6,r8__condX2 <= 0
 c202:  - aux_c_b0,r10 + aux_cap_b0,r10,band_LOW_PAIRED
        + aux_cap_b0,r10,band_HIGH_PAIRED + aux_cap_b0,r10,band_UNPAIRED  = 0
 c203:  aux_cap_helper__b5,r5,band_UNPAIRED__Z1
        + aux_cap_helper__b5,r5,band_UNPAIRED__Z2  = 1
 c204:  - 6.36037231620095 sv_function_helpervar_b0,r10__Z2
        + sv_function_helpervar_b0,r10__condX2 <= 0
 c205:  sv_function_helpervar_b2,r2__condX3
        + sv_function_helpervar_b2,r2__condX2
        + sv_function_helpervar_b2,r2__condX1 - aux_c_b2,r2  = 0
 c206:  - aux_cap_helper__b4,r3,band_UNPAIRED__Z1
        + aux_cap_helper__b4,r3,band_UNPAIRED__condX1 <= 0
 c207:  - aux_cap_helper__b0,r5,band_HIGH_PAIRED__condX1 <= 0
 c208:  - 0.253841888 aux_cap_helper__b8,r5,band_UNPAIRED__Z2
        + 0.888446608 aux_cap_helper__b8,r5,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b8,r5,band_UNPAIRED__condX1
        - aux_cap_b8,r5,band_UNPAIRED  = 0
 c209:  0.888446608 aux_cap_helper__b0,r6,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b0,r6,band_UNPAIRED__condX1
        - 0.253841888 aux_cap_helper__b0,r6,band_UNPAIRED__Z2
        - aux_cap_b0,r6,band_UNPAIRED  = 0
 c210:  - aux_c_b0,r8 + sv_function_helpervar_b0,r8__condX1
        + sv_function_helpervar_b0,r8__condX2
        + sv_function_helpervar_b0,r8__condX3  = 0
 c211:  aux_cap_helper__b2,r1,band_UNPAIRED__Z2
        - aux_cap_helper__b2,r1,band_UNPAIRED__condX2 <= 0
 c212:  aux_cap_helper__b3,r4,band_UNPAIRED__condX1
        - aux_cap_helper__b3,r4,band_UNPAIRED__Z1 <= 0
 c213:  - X__b4,r2,band_HIGH_PAIRED
        + aux_cap_helper__b4,r2,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b4,r2,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b4,r2,band_HIGH_PAIRED__condX1  = 0
 c214:  - aux_cap_helper__b9,r13,band_UNPAIRED__condX1 <= 0
 c215:  aux_cap_b3,r8,band_HIGH_PAIRED + aux_cap_b3,r8,band_LOW_PAIRED
        - aux_c_b3,r8 + aux_cap_b3,r8,band_UNPAIRED  = 0
 c216:  6.53397023325115 sv_function_helpervar_b0,r4__Z3
        - sv_function_helpervar_b0,r4__condX3 <= 0
 c217:  aux_cap_helper__b0,r11,band_LOW_PAIRED__condX1
        - aux_cap_helper__b0,r11,band_LOW_PAIRED__Z1 <= 0
 c218:  - aux_Omega_b6,r11 + 1184.74227103159 aux_sv_b6,r11  = 0
 c219:  - aux_cap_helper__b0,r7,band_UNPAIRED__condX1 <= 0
 c220:  - aux_cap_helper__b3,r2,band_HIGH_PAIRED__condX1 <= 0
 c221:  44.0777559238 sv_function_helpervar_b6,r6__condX2
        + 942.4308565915 sv_function_helpervar_b6,r6__condX1
        + 15.0613159016 sv_function_helpervar_b6,r6__condX3
        + 142.346821938974 sv_function_helpervar_b6,r6__Z2
        + 314.426868537727 sv_function_helpervar_b6,r6__Z3 - aux_sv_b6,r6  = 0
 c222:  sv_function_helpervar_b4,r2__condX1
        + sv_function_helpervar_b4,r2__condX2
        + sv_function_helpervar_b4,r2__condX3 - aux_c_b4,r2  = 0
 c223:  - 1.2715511236 aux_cap_helper__b5,r5,band_LOW_PAIRED__Z2
        + 4.4504289328 aux_cap_helper__b5,r5,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b5,r5,band_LOW_PAIRED__condX1
        - aux_cap_b5,r5,band_LOW_PAIRED  = 0
 c224:  aux_cap_helper__b6,r0,band_UNPAIRED__Z2
        - aux_cap_helper__b6,r0,band_UNPAIRED__condX2 <= 0
 c225:  aux_cap_helper__b5,r11,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b5,r11,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b5,r11,band_HIGH_PAIRED__Z1  = 1
 c226:  4.4504289328 aux_cap_helper__b0,r7,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b0,r7,band_LOW_PAIRED__condX1
        - aux_cap_b0,r7,band_LOW_PAIRED
        - 1.2715511236 aux_cap_helper__b0,r7,band_LOW_PAIRED__Z2  = 0
 c227:  sv_function_helpervar_b5,r5__condX1
        - 0.158453086904847 sv_function_helpervar_b5,r5__Z1 <= 0
 c228:  aux_cap_helper__b0,r11,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b0,r11,band_LOW_PAIRED__Z2 <= 0
 c229:  - aux_cap_b6,r8,band_LOW_PAIRED
        + 3.1788778092 aux_cap_helper__b6,r8,band_LOW_PAIRED__condX1
        + 4.4504289328 aux_cap_helper__b6,r8,band_LOW_PAIRED__condX2
        - 1.2715511236 aux_cap_helper__b6,r8,band_LOW_PAIRED__Z2  = 0
 c230:  - aux_cap_helper__b0,r2,band_LOW_PAIRED__Z1
        + aux_cap_helper__b0,r2,band_LOW_PAIRED__condX1 <= 0
 c231:  - aux_cap_helper__b0,r2,band_HIGH_PAIRED__condX1 <= 0
 c232:  aux_cap_b4,r13,band_UNPAIRED - aux_c_b4,r13
        + aux_cap_b4,r13,band_LOW_PAIRED + aux_cap_b4,r13,band_HIGH_PAIRED  = 0
 c233:  - aux_cap_helper__b5,r1,band_UNPAIRED__condX2
        + aux_cap_helper__b5,r1,band_UNPAIRED__Z2 <= 0
 c234:  - aux_cap_helper__b7,r1,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b7,r1,band_HIGH_PAIRED__condX1 <= 0
 c235:  sv_function_helpervar_b0,r10__Z2 + sv_function_helpervar_b0,r10__Z3
        + sv_function_helpervar_b0,r10__Z1  = 1
 c236:  0.0102040816326531 X__b7,r4,band_LOW_PAIRED
        + 0.0102040816326531 X__b7,r4,band_UNPAIRED
        + 0.0102040816326531 X__b7,r4,band_HIGH_PAIRED
        - GlobalBidder_W_ir__b7,r4 <= 0
 c237:  aux_cap_helper__b0,r4,band_UNPAIRED__condX1
        - aux_cap_helper__b0,r4,band_UNPAIRED__Z1 <= 0
 c238:  - aux_sv_b5,r4 + 15.0714278995 sv_function_helpervar_b5,r4__condX3
        + 48.9809044585 sv_function_helpervar_b5,r4__condX2
        + 978.9018729809 sv_function_helpervar_b5,r4__condX1
        + 335.670141293665 sv_function_helpervar_b5,r4__Z3
        + 147.348848048816 sv_function_helpervar_b5,r4__Z2  = 0
 c239:  - aux_cap_helper__b6,r3,band_LOW_PAIRED__condX1 <= 0
 c240:  aux_cap_helper__b4,r1,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b4,r1,band_HIGH_PAIRED__Z1 <= 0
 c241:  sv_function_helpervar_b7,r7__condX2
        + sv_function_helpervar_b7,r7__condX1
        + sv_function_helpervar_b7,r7__condX3 - aux_c_b7,r7  = 0
 c242:  - 2 aux_cap_helper__b0,r2,band_LOW_PAIRED__Z2
        + aux_cap_helper__b0,r2,band_LOW_PAIRED__condX2 <= 0
 c243:  - aux_cap_helper__b1,r7,band_LOW_PAIRED__condX1 <= 0
 c244:  - sv_function_helpervar_b6,r9__condX3
        + 5.57476174070583 sv_function_helpervar_b6,r9__Z3 <= 0
 c245:  aux_cap_helper__b2,r4,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b2,r4,band_UNPAIRED__Z2 <= 0
 c246:  sv_function_helpervar_b4,r5__condX2
        + sv_function_helpervar_b4,r5__condX3
        + sv_function_helpervar_b4,r5__condX1 - aux_c_b4,r5  = 0
 c247:  0.0102040816326531 X__b7,r10,band_HIGH_PAIRED
        + 0.0102040816326531 X__b7,r10,band_UNPAIRED
        + 0.0102040816326531 X__b7,r10,band_LOW_PAIRED
        - GlobalBidder_W_ir__b7,r10 <= 0
 c248:  - 2 aux_cap_helper__b9,r2,band_UNPAIRED__Z2
        + aux_cap_helper__b9,r2,band_UNPAIRED__condX2 <= 0
 c249:  aux_cap_b0,r5,band_LOW_PAIRED - aux_c_b0,r5
        + aux_cap_b0,r5,band_UNPAIRED + aux_cap_b0,r5,band_HIGH_PAIRED  = 0
 c250:  sv_function_helpervar_b1,r2__Z1 + sv_function_helpervar_b1,r2__Z3
        + sv_function_helpervar_b1,r2__Z2  = 1
 c251:  133.82176750906 sv_function_helpervar_b4,r1__Z2
        + 286.151628374986 sv_function_helpervar_b4,r1__Z3 - aux_sv_b4,r1
        + 883.4028075217 sv_function_helpervar_b4,r1__condX1
        + 38.8514636759 sv_function_helpervar_b4,r1__condX2
        + 14.6595468568 sv_function_helpervar_b4,r1__condX3  = 0
 c252:  - 15.8453086887446 sv_function_helpervar_b0,r13__Z3
        + sv_function_helpervar_b0,r13__condX3 <= 0
 c253:  aux_cap_helper__b5,r9,band_UNPAIRED__condX2
        + aux_cap_helper__b5,r9,band_UNPAIRED__condX1 - X__b5,r9,band_UNPAIRED
         = 0
 c254:  - X__b7,r11,band_UNPAIRED
        + aux_cap_helper__b7,r11,band_UNPAIRED__condX1
        + aux_cap_helper__b7,r11,band_UNPAIRED__condX2  = 0
 c255:  - 0.158453086963894 sv_function_helpervar_b1,r13__Z1
        + sv_function_helpervar_b1,r13__condX1 <= 0
 c256:  - 0.15845308689699 sv_function_helpervar_b2,r12__Z1
        + sv_function_helpervar_b2,r12__condX1 <= 0
 c257:  - aux_cap_helper__b4,r4,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b4,r4,band_HIGH_PAIRED__condX1 <= 0
 c258:  - sv_function_helpervar_b0,r12__condX1 <= 0
 c259:  sv_function_helpervar_b9,r7__condX3
        + sv_function_helpervar_b9,r7__condX2
        + sv_function_helpervar_b9,r7__condX1 - aux_c_b9,r7  = 0
 c260:  aux_Omega_b7,r10 + aux_Omega_b7,r11 + aux_Omega_b7,r12
        + aux_Omega_b7,r13 + aux_Omega_b7,r0 + aux_Omega_b7,r1
        + aux_Omega_b7,r2 + aux_Omega_b7,r3 + aux_Omega_b7,r4 + aux_Omega_b7,r5
        + aux_Omega_b7,r6 + aux_Omega_b7,r7 + aux_Omega_b7,r8 + aux_Omega_b7,r9
        - GlobalBidder_PSI__i7,k0 - 20668026.8748422 GlobalBidder_hat{W}__i7,k0
        >= -20668026.8748422
 c261:  sv_function_helpervar_b6,r8__Z2 + sv_function_helpervar_b6,r8__Z1
        + sv_function_helpervar_b6,r8__Z3  = 1
 c262:  - aux_sv_b7,r9 + 190.886407036109 sv_function_helpervar_b7,r9__Z2
        + 388.081291685134 sv_function_helpervar_b7,r9__Z3
        + 1255.669439237 sv_function_helpervar_b7,r9__condX1
        + 50.9822309085 sv_function_helpervar_b7,r9__condX2
        + 22.0144023327 sv_function_helpervar_b7,r9__condX3  = 0
 c263:  aux_cap_helper__b8,r7,band_LOW_PAIRED__Z2
        - aux_cap_helper__b8,r7,band_LOW_PAIRED__condX2 <= 0
 c264:  - X__b9,r3,band_UNPAIRED + aux_cap_helper__b9,r3,band_UNPAIRED__condX2
        + aux_cap_helper__b9,r3,band_UNPAIRED__condX1  = 0
 c265:  aux_cap_helper__b9,r11,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b9,r11,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b9,r11,band_HIGH_PAIRED__condX3
        - X__b9,r11,band_HIGH_PAIRED  = 0
 c266:  aux_cap_helper__b0,r13,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b0,r13,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b0,r13,band_HIGH_PAIRED__condX3
        - X__b0,r13,band_HIGH_PAIRED  = 0
 c267:  1.21690011250031 aux_cap_helper__b6,r3,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b6,r3,band_HIGH_PAIRED__Z2
        + 1.8253501688 aux_cap_helper__b6,r3,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b6,r3,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b6,r3,band_HIGH_PAIRED__condX1
        - aux_cap_b6,r3,band_HIGH_PAIRED  = 0
 c268:  - X__b4,r12,band_LOW_PAIRED
        + aux_cap_helper__b4,r12,band_LOW_PAIRED__condX2
        + aux_cap_helper__b4,r12,band_LOW_PAIRED__condX1  = 0
 c269:  - aux_cap_helper__b7,r9,band_LOW_PAIRED__condX1 <= 0
 c270:  sv_function_helpervar_b0,r3__condX2
        - 6.31602055756089 sv_function_helpervar_b0,r3__Z2 <= 0
 c271:  - 2 aux_cap_helper__b4,r4,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b4,r4,band_HIGH_PAIRED__condX2 <= 0
 c272:  - sv_function_helpervar_b6,r11__condX2
        + 0.15845308690825 sv_function_helpervar_b6,r11__Z2 <= 0
 c273:  - X__b7,r9,band_HIGH_PAIRED
        + aux_cap_helper__b7,r9,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b7,r9,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b7,r9,band_HIGH_PAIRED__condX1  = 0
 c274:  - aux_Omega_b7,r5 + 1407.65776572845 aux_sv_b7,r5  = 0
 c275:  14.6527541433 sv_function_helpervar_b6,r11__condX3
        + 942.4308566218 sv_function_helpervar_b6,r11__condX1
        + 46.2951154351 sv_function_helpervar_b6,r11__condX2 - aux_sv_b6,r11
        + 320.900655716868 sv_function_helpervar_b6,r11__Z3
        + 141.995474479845 sv_function_helpervar_b6,r11__Z2  = 0
 c276:  aux_cap_helper__b9,r9,band_LOW_PAIRED__condX1
        - aux_cap_helper__b9,r9,band_LOW_PAIRED__Z1 <= 0
 c277:  sv_function_helpervar_b5,r10__condX3
        - 15.8453086887446 sv_function_helpervar_b5,r10__Z3 <= 0
 c278:  0.63460472 aux_cap_helper__b5,r7,band_UNPAIRED__condX1
        + 0.888446608 aux_cap_helper__b5,r7,band_UNPAIRED__condX2
        - aux_cap_b5,r7,band_UNPAIRED
        - 0.253841888 aux_cap_helper__b5,r7,band_UNPAIRED__Z2  = 0
 c279:  - aux_cap_helper__b9,r0,band_LOW_PAIRED__Z1
        + aux_cap_helper__b9,r0,band_LOW_PAIRED__condX1 <= 0
 c280:  - 1.2715511236 aux_cap_helper__b8,r2,band_LOW_PAIRED__Z2
        - aux_cap_b8,r2,band_LOW_PAIRED
        + 4.4504289328 aux_cap_helper__b8,r2,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b8,r2,band_LOW_PAIRED__condX1  = 0
 c281:  - aux_cap_helper__b9,r0,band_UNPAIRED__condX2
        + aux_cap_helper__b9,r0,band_UNPAIRED__Z2 <= 0
 c282:  - aux_cap_b4,r11,band_LOW_PAIRED
        - 1.2715511236 aux_cap_helper__b4,r11,band_LOW_PAIRED__Z2
        + 3.1788778092 aux_cap_helper__b4,r11,band_LOW_PAIRED__condX1
        + 4.4504289328 aux_cap_helper__b4,r11,band_LOW_PAIRED__condX2  = 0
 c283:  - aux_cap_b2,r5,band_HIGH_PAIRED
        + 2.0281668543 aux_cap_helper__b2,r5,band_HIGH_PAIRED__condX1
        + 2.8394335959 aux_cap_helper__b2,r5,band_HIGH_PAIRED__condX2
        + 1.8253501688 aux_cap_helper__b2,r5,band_HIGH_PAIRED__condX3
        + 1.21690011250031 aux_cap_helper__b2,r5,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b2,r5,band_HIGH_PAIRED__Z2  = 0
 c284:  2 aux_cap_helper__b5,r9,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b5,r9,band_HIGH_PAIRED__condX3 <= 0
 c285:  aux_cap_helper__b3,r13,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b3,r13,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b3,r13,band_HIGH_PAIRED__condX3
        - X__b3,r13,band_HIGH_PAIRED  = 0
 c286:  - aux_cap_helper__b8,r12,band_LOW_PAIRED__condX1 <= 0
 c287:  aux_cap_helper__b6,r2,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b6,r2,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b6,r2,band_HIGH_PAIRED__Z1  = 1
 c288:  aux_cap_helper__b4,r1,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b4,r1,band_HIGH_PAIRED__Z2 <= 0
 c289:  - aux_cap_helper__b2,r9,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b2,r9,band_HIGH_PAIRED__Z3 <= 0
 c290:  - X__b8,r2,band_LOW_PAIRED
        + aux_cap_helper__b8,r2,band_LOW_PAIRED__condX2
        + aux_cap_helper__b8,r2,band_LOW_PAIRED__condX1  = 0
 c291:  - 2 aux_cap_helper__b9,r0,band_LOW_PAIRED__Z2
        + aux_cap_helper__b9,r0,band_LOW_PAIRED__condX2 <= 0
 c292:  - aux_cap_helper__b3,r10,band_UNPAIRED__condX1 <= 0
 c293:  - aux_cap_helper__b5,r6,band_UNPAIRED__condX1 <= 0
 c294:  - 2 aux_cap_helper__b4,r3,band_UNPAIRED__Z2
        + aux_cap_helper__b4,r3,band_UNPAIRED__condX2 <= 0
 c295:  0.0102040816326531 X__b9,r5,band_UNPAIRED
        + 0.0102040816326531 X__b9,r5,band_HIGH_PAIRED
        + 0.0102040816326531 X__b9,r5,band_LOW_PAIRED
        - GlobalBidder_W_ir__b9,r5 <= 0
 c296:  aux_cap_helper__b5,r6,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b5,r6,band_HIGH_PAIRED__condX2 <= 0
 c297:  X__b7,r2,band_HIGH_PAIRED - GlobalBidder_W_ir__b7,r2
        + X__b7,r2,band_LOW_PAIRED + X__b7,r2,band_UNPAIRED >= 0
 c298:  - aux_cap_helper__b0,r10,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b0,r10,band_HIGH_PAIRED__Z2 <= 0
 c299:  - aux_cap_helper__b8,r0,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b8,r0,band_HIGH_PAIRED__Z3 <= 0
 c300:  sv_function_helpervar_b2,r5__condX2
        + sv_function_helpervar_b2,r5__condX3
        + sv_function_helpervar_b2,r5__condX1 - aux_c_b2,r5  = 0
 c301:  - aux_cap_helper__b7,r3,band_UNPAIRED__Z1
        + aux_cap_helper__b7,r3,band_UNPAIRED__condX1 <= 0
 c302:  - sv_function_helpervar_b9,r5__condX1 <= 0
 c303:  aux_cap_b6,r7,band_HIGH_PAIRED + aux_cap_b6,r7,band_UNPAIRED
        - aux_c_b6,r7 + aux_cap_b6,r7,band_LOW_PAIRED  = 0
 c304:  sv_function_helpervar_b9,r1__condX1
        + sv_function_helpervar_b9,r1__condX2
        + sv_function_helpervar_b9,r1__condX3 - aux_c_b9,r1  = 0
 c305:  X__b7,r12,band_HIGH_PAIRED + X__b7,r12,band_UNPAIRED
        + X__b7,r12,band_LOW_PAIRED - GlobalBidder_W_ir__b7,r12 >= 0
 c306:  - aux_cap_helper__b7,r1,band_LOW_PAIRED__condX2
        + aux_cap_helper__b7,r1,band_LOW_PAIRED__Z2 <= 0
 c307:  sv_function_helpervar_b4,r5__condX3
        - 15.8453086887446 sv_function_helpervar_b4,r5__Z3 <= 0
 c308:  2 aux_cap_helper__b5,r6,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b5,r6,band_HIGH_PAIRED__condX3 <= 0
 c309:  aux_cap_helper__b6,r10,band_LOW_PAIRED__Z2
        + aux_cap_helper__b6,r10,band_LOW_PAIRED__Z1  = 1
 c310:  - X__b1,r4,band_UNPAIRED + aux_cap_helper__b1,r4,band_UNPAIRED__condX1
        + aux_cap_helper__b1,r4,band_UNPAIRED__condX2  = 0
 c311:  1.8253501688 aux_cap_helper__b4,r9,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b4,r9,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b4,r9,band_HIGH_PAIRED__condX1
        - 0.8112667416 aux_cap_helper__b4,r9,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b4,r9,band_HIGH_PAIRED__Z3
        - aux_cap_b4,r9,band_HIGH_PAIRED  = 0
 c312:  - aux_cap_helper__b0,r10,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b0,r10,band_HIGH_PAIRED__Z3 <= 0
 c313:  - 2 aux_cap_helper__b7,r3,band_UNPAIRED__Z2
        + aux_cap_helper__b7,r3,band_UNPAIRED__condX2 <= 0
 c314:  - sv_function_helpervar_b4,r4__condX1 <= 0
 c315:  - aux_cap_helper__b2,r6,band_UNPAIRED__condX1 <= 0
 c316:  aux_cap_helper__b5,r9,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b5,r9,band_HIGH_PAIRED__condX2 <= 0
 c317:  - aux_c_b2,r8 + sv_function_helpervar_b2,r8__condX1
        + sv_function_helpervar_b2,r8__condX2
        + sv_function_helpervar_b2,r8__condX3  = 0
 c318:  sv_function_helpervar_b9,r4__condX1
        + sv_function_helpervar_b9,r4__condX3
        + sv_function_helpervar_b9,r4__condX2 - aux_c_b9,r4  = 0
 c319:  - aux_cap_helper__b0,r13,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b0,r13,band_HIGH_PAIRED__Z2 <= 0
 c320:  - aux_cap_helper__b5,r12,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b5,r12,band_HIGH_PAIRED__Z2 <= 0
 c321:  - aux_cap_helper__b8,r3,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b8,r3,band_HIGH_PAIRED__Z3 <= 0
 c322:  sv_function_helpervar_b9,r6__condX3
        - 15.8453086887446 sv_function_helpervar_b9,r6__Z3 <= 0
 c323:  aux_cap_helper__b4,r5,band_LOW_PAIRED__condX2
        + aux_cap_helper__b4,r5,band_LOW_PAIRED__condX1
        - X__b4,r5,band_LOW_PAIRED  = 0
 c324:  - aux_cap_helper__b2,r9,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b2,r9,band_HIGH_PAIRED__Z2 <= 0
 c325:  - X__b5,r0,band_HIGH_PAIRED
        + aux_cap_helper__b5,r0,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b5,r0,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b5,r0,band_HIGH_PAIRED__condX3  = 0
 c326:  - aux_cap_helper__b0,r2,band_UNPAIRED__condX2
        + aux_cap_helper__b0,r2,band_UNPAIRED__Z2 <= 0
 c327:  - aux_cap_helper__b7,r6,band_UNPAIRED__condX1 <= 0
 c328:  sv_function_helpervar_b7,r9__Z1 + sv_function_helpervar_b7,r9__Z2
        + sv_function_helpervar_b7,r9__Z3  = 1
 c329:  - 3 aux_cap_helper__b8,r6,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b8,r6,band_HIGH_PAIRED__condX3 <= 0
 c330:  aux_cap_helper__b6,r13,band_LOW_PAIRED__condX1
        + aux_cap_helper__b6,r13,band_LOW_PAIRED__condX2
        - X__b6,r13,band_LOW_PAIRED  = 0
 c331:  - aux_cap_helper__b7,r9,band_HIGH_PAIRED__condX1 <= 0
 c332:  - aux_cap_b2,r9,band_UNPAIRED
        - 0.253841888 aux_cap_helper__b2,r9,band_UNPAIRED__Z2
        + 0.888446608 aux_cap_helper__b2,r9,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b2,r9,band_UNPAIRED__condX1  = 0
 c333:  - aux_sv_b2,r7 + 17.2912055910862 sv_function_helpervar_b2,r7__Z2
        + 38.9419979789562 sv_function_helpervar_b2,r7__Z3
        + 1.7911306374 sv_function_helpervar_b2,r7__condX3
        + 5.5916151604 sv_function_helpervar_b2,r7__condX2
        + 114.7166939468 sv_function_helpervar_b2,r7__condX1  = 0
 c334:  sv_function_helpervar_b4,r8__condX3
        + sv_function_helpervar_b4,r8__condX2
        + sv_function_helpervar_b4,r8__condX1 - aux_c_b4,r8  = 0
 c335:  142.627160508292 sv_function_helpervar_b6,r0__Z2
        + 308.472186578231 sv_function_helpervar_b6,r0__Z3
        + 15.4371168454 sv_function_helpervar_b6,r0__condX3
        + 42.3085346637 sv_function_helpervar_b6,r0__condX2
        + 942.430856482 sv_function_helpervar_b6,r0__condX1 - aux_sv_b6,r0  = 0
 c336:  aux_cap_b9,r6,band_HIGH_PAIRED + aux_cap_b9,r6,band_LOW_PAIRED
        + aux_cap_b9,r6,band_UNPAIRED - aux_c_b9,r6  = 0
 c337:  - sv_function_helpervar_b9,r13__condX1 <= 0
 c338:  sv_function_helpervar_b8,r5__Z1 + sv_function_helpervar_b8,r5__Z3
        + sv_function_helpervar_b8,r5__Z2  = 1
 c339:  - 0.8112667416 aux_cap_helper__b0,r0,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b0,r0,band_HIGH_PAIRED__Z3
        + 1.8253501688 aux_cap_helper__b0,r0,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b0,r0,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b0,r0,band_HIGH_PAIRED__condX1
        - aux_cap_b0,r0,band_HIGH_PAIRED  = 0
 c340:  - aux_cap_helper__b8,r10,band_UNPAIRED__condX1 <= 0
 c341:  - aux_cap_helper__b4,r12,band_UNPAIRED__Z1
        + aux_cap_helper__b4,r12,band_UNPAIRED__condX1 <= 0
 c342:  aux_cap_helper__b2,r13,band_UNPAIRED__condX1
        - aux_cap_helper__b2,r13,band_UNPAIRED__Z1 <= 0
 c343:  - aux_cap_b8,r2,band_HIGH_PAIRED
        - 0.8112667416 aux_cap_helper__b8,r2,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b8,r2,band_HIGH_PAIRED__Z3
        + 1.8253501688 aux_cap_helper__b8,r2,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b8,r2,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b8,r2,band_HIGH_PAIRED__condX1  = 0
 c344:  2.8394335959 aux_cap_helper__b2,r1,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b2,r1,band_HIGH_PAIRED__condX1
        + 1.8253501688 aux_cap_helper__b2,r1,band_HIGH_PAIRED__condX3
        - aux_cap_b2,r1,band_HIGH_PAIRED
        - 0.8112667416 aux_cap_helper__b2,r1,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b2,r1,band_HIGH_PAIRED__Z3  = 0
 c345:  - 15.8453086887446 sv_function_helpervar_b3,r6__Z3
        + sv_function_helpervar_b3,r6__condX3 <= 0
 c346:  - 2 aux_cap_helper__b1,r4,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b1,r4,band_HIGH_PAIRED__condX2 <= 0
 c347:  aux_cap_helper__b2,r1,band_LOW_PAIRED__condX2
        + aux_cap_helper__b2,r1,band_LOW_PAIRED__condX1
        - X__b2,r1,band_LOW_PAIRED  = 0
 c348:  aux_cap_helper__b2,r13,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b2,r13,band_UNPAIRED__Z2 <= 0
 c349:  aux_cap_helper__b4,r4,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b4,r4,band_UNPAIRED__Z2 <= 0
 c350:  - aux_cap_helper__b3,r8,band_LOW_PAIRED__condX2
        + aux_cap_helper__b3,r8,band_LOW_PAIRED__Z2 <= 0
 c351:  aux_cap_helper__b3,r10,band_LOW_PAIRED__condX2
        + aux_cap_helper__b3,r10,band_LOW_PAIRED__condX1
        - X__b3,r10,band_LOW_PAIRED  = 0
 c352:  aux_cap_b9,r10,band_HIGH_PAIRED + aux_cap_b9,r10,band_UNPAIRED
        - aux_c_b9,r10 + aux_cap_b9,r10,band_LOW_PAIRED  = 0
 c353:  sv_function_helpervar_b0,r5__condX1
        + sv_function_helpervar_b0,r5__condX3
        + sv_function_helpervar_b0,r5__condX2 - aux_c_b0,r5  = 0
 c354:  aux_cap_b8,r0,band_HIGH_PAIRED - aux_c_b8,r0
        + aux_cap_b8,r0,band_LOW_PAIRED + aux_cap_b8,r0,band_UNPAIRED  = 0
 c355:  6.80737543489106 sv_function_helpervar_b7,r9__Z3
        - sv_function_helpervar_b7,r9__condX3 <= 0
 c356:  aux_cap_helper__b6,r4,band_UNPAIRED__condX2
        + aux_cap_helper__b6,r4,band_UNPAIRED__condX1 - X__b6,r4,band_UNPAIRED
         = 0
 c357:  - sv_function_helpervar_b8,r7__condX3
        + 6.83798725563921 sv_function_helpervar_b8,r7__Z3 <= 0
 c358:  - 2 aux_cap_helper__b4,r7,band_LOW_PAIRED__Z2
        + aux_cap_helper__b4,r7,band_LOW_PAIRED__condX2 <= 0
 c359:  22.0089645944 sv_function_helpervar_b7,r12__condX3
        + 50.9993586957 sv_function_helpervar_b7,r12__condX2
        + 1255.6694399323 sv_function_helpervar_b7,r12__condX1
        + 388.167454327412 sv_function_helpervar_b7,r12__Z3
        + 190.883693089824 sv_function_helpervar_b7,r12__Z2 - aux_sv_b7,r12
         = 0
 c360:  aux_cap_helper__b5,r0,band_UNPAIRED__Z2
        - aux_cap_helper__b5,r0,band_UNPAIRED__condX2 <= 0
 c361:  aux_cap_b8,r8,band_LOW_PAIRED - aux_c_b8,r8
        + aux_cap_b8,r8,band_HIGH_PAIRED + aux_cap_b8,r8,band_UNPAIRED  = 0
 c362:  - aux_sv_b5,r2 + 330.14514171018 sv_function_helpervar_b5,r2__Z3
        + 147.669777561709 sv_function_helpervar_b5,r2__Z2
        + 46.9555130946 sv_function_helpervar_b5,r2__condX2
        + 15.4201115189 sv_function_helpervar_b5,r2__condX3
        + 978.9018728594 sv_function_helpervar_b5,r2__condX1  = 0
 c363:  aux_cap_helper__b8,r3,band_LOW_PAIRED__condX1
        - aux_cap_helper__b8,r3,band_LOW_PAIRED__Z1 <= 0
 c364:  sv_function_helpervar_b6,r12__Z3 + sv_function_helpervar_b6,r12__Z2
        + sv_function_helpervar_b6,r12__Z1  = 1
 c365:  - X__b6,r2,band_UNPAIRED + aux_cap_helper__b6,r2,band_UNPAIRED__condX1
        + aux_cap_helper__b6,r2,band_UNPAIRED__condX2  = 0
 c366:  - aux_cap_helper__b9,r2,band_LOW_PAIRED__condX1 <= 0
 c367:  - aux_cap_helper__b8,r0,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b8,r0,band_HIGH_PAIRED__Z2 <= 0
 c368:  - aux_cap_helper__b6,r11,band_HIGH_PAIRED__condX1 <= 0
 c369:  - aux_cap_b4,r2,band_HIGH_PAIRED
        + 1.21690011250031 aux_cap_helper__b4,r2,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b4,r2,band_HIGH_PAIRED__Z2
        + 2.8394335959 aux_cap_helper__b4,r2,band_HIGH_PAIRED__condX2
        + 1.8253501688 aux_cap_helper__b4,r2,band_HIGH_PAIRED__condX3
        + 2.0281668543 aux_cap_helper__b4,r2,band_HIGH_PAIRED__condX1  = 0
 c370:  - aux_cap_helper__b1,r6,band_UNPAIRED__condX1 <= 0
 c371:  0.0102040816326531 X__b9,r10,band_LOW_PAIRED
        + 0.0102040816326531 X__b9,r10,band_UNPAIRED
        - GlobalBidder_W_ir__b9,r10
        + 0.0102040816326531 X__b9,r10,band_HIGH_PAIRED <= 0
 c372:  - 0.8112667416 aux_cap_helper__b1,r10,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b1,r10,band_HIGH_PAIRED__Z3
        - aux_cap_b1,r10,band_HIGH_PAIRED
        + 1.8253501688 aux_cap_helper__b1,r10,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b1,r10,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b1,r10,band_HIGH_PAIRED__condX1  = 0
 c373:  aux_cap_b7,r2,band_LOW_PAIRED - aux_c_b7,r2
        + aux_cap_b7,r2,band_HIGH_PAIRED + aux_cap_b7,r2,band_UNPAIRED  = 0
 c374:  - aux_cap_b6,r11,band_HIGH_PAIRED
        + 1.8253501688 aux_cap_helper__b6,r11,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b6,r11,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b6,r11,band_HIGH_PAIRED__condX1
        - 0.8112667416 aux_cap_helper__b6,r11,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b6,r11,band_HIGH_PAIRED__Z3  = 0
 c375:  - 0.158453086894059 sv_function_helpervar_b1,r5__Z1
        + sv_function_helpervar_b1,r5__condX1 <= 0
 c376:  aux_cap_helper__b8,r3,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b8,r3,band_LOW_PAIRED__Z2 <= 0
 c377:  - aux_Omega_b4,r0 + 107.358593990881 aux_sv_b4,r0  = 0
 c378:  - aux_cap_helper__b1,r2,band_UNPAIRED__condX2
        + aux_cap_helper__b1,r2,band_UNPAIRED__Z2 <= 0
 c379:  aux_cap_helper__b9,r6,band_LOW_PAIRED__Z2
        + aux_cap_helper__b9,r6,band_LOW_PAIRED__Z1  = 1
 c380:  - X__b2,r8,band_LOW_PAIRED
        + aux_cap_helper__b2,r8,band_LOW_PAIRED__condX1
        + aux_cap_helper__b2,r8,band_LOW_PAIRED__condX2  = 0
 c381:  - aux_cap_helper__b1,r4,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b1,r4,band_HIGH_PAIRED__condX1 <= 0
 c382:  - 3 aux_cap_helper__b5,r13,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b5,r13,band_HIGH_PAIRED__condX3 <= 0
 c383:  aux_cap_helper__b3,r9,band_UNPAIRED__Z1
        + aux_cap_helper__b3,r9,band_UNPAIRED__Z2  = 1
 c384:  - aux_cap_helper__b7,r3,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b7,r3,band_HIGH_PAIRED__condX1 <= 0
 c385:  - sv_function_helpervar_b6,r12__condX2
        + 0.158453086892631 sv_function_helpervar_b6,r12__Z2 <= 0
 c386:  - 0.158453086984489 sv_function_helpervar_b3,r1__Z1
        + sv_function_helpervar_b3,r1__condX1 <= 0
 c387:  - 5.6968505612505 sv_function_helpervar_b2,r7__Z2
        + sv_function_helpervar_b2,r7__condX2 <= 0
 c388:  - 1.2715511236 aux_cap_helper__b0,r13,band_LOW_PAIRED__Z2
        - aux_cap_b0,r13,band_LOW_PAIRED
        + 3.1788778092 aux_cap_helper__b0,r13,band_LOW_PAIRED__condX1
        + 4.4504289328 aux_cap_helper__b0,r13,band_LOW_PAIRED__condX2  = 0
 c389:  sv_function_helpervar_b1,r7__condX1
        - 0.158453086922884 sv_function_helpervar_b1,r7__Z1 <= 0
 c390:  - 5.43751382574535 sv_function_helpervar_b3,r5__Z2
        + sv_function_helpervar_b3,r5__condX2 <= 0
 c391:  aux_cap_helper__b1,r3,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b1,r3,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b1,r3,band_HIGH_PAIRED__Z1  = 1
 c392:  aux_cap_helper__b7,r2,band_UNPAIRED__condX1
        - aux_cap_helper__b7,r2,band_UNPAIRED__Z1 <= 0
 c393:  - aux_cap_helper__b4,r7,band_LOW_PAIRED__Z1
        + aux_cap_helper__b4,r7,band_LOW_PAIRED__condX1 <= 0
 c394:  - 2 aux_cap_helper__b4,r12,band_UNPAIRED__Z2
        + aux_cap_helper__b4,r12,band_UNPAIRED__condX2 <= 0
 c395:  sv_function_helpervar_b9,r12__condX2
        + sv_function_helpervar_b9,r12__condX1
        + sv_function_helpervar_b9,r12__condX3 - aux_c_b9,r12  = 0
 c396:  - 1.2715511236 aux_cap_helper__b4,r7,band_LOW_PAIRED__Z2
        + 4.4504289328 aux_cap_helper__b4,r7,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b4,r7,band_LOW_PAIRED__condX1
        - aux_cap_b4,r7,band_LOW_PAIRED  = 0
 c397:  - 2 aux_cap_helper__b7,r3,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b7,r3,band_HIGH_PAIRED__condX2 <= 0
 c398:  14 GlobalBidder_hat{W}__i8,k2 + GlobalBidder_W_i_8 <= 26
 c399:  aux_cap_helper__b9,r7,band_LOW_PAIRED__Z2
        - aux_cap_helper__b9,r7,band_LOW_PAIRED__condX2 <= 0
 c400:  - sv_function_helpervar_b9,r6__condX1 <= 0
 c401:  aux_cap_helper__b4,r0,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b4,r0,band_HIGH_PAIRED__Z3 <= 0
 c402:  4.4504289328 aux_cap_helper__b9,r5,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b9,r5,band_LOW_PAIRED__condX1
        - 1.2715511236 aux_cap_helper__b9,r5,band_LOW_PAIRED__Z2
        - aux_cap_b9,r5,band_LOW_PAIRED  = 0
 c403:  aux_Omega_b9,r1 + aux_Omega_b9,r0 + aux_Omega_b9,r3 + aux_Omega_b9,r2
        + aux_Omega_b9,r5 + aux_Omega_b9,r4 + aux_Omega_b9,r7 + aux_Omega_b9,r6
        + aux_Omega_b9,r9 + aux_Omega_b9,r8 - GlobalBidder_PSI__i9,k0
        + aux_Omega_b9,r13 + aux_Omega_b9,r10 + aux_Omega_b9,r11
        + aux_Omega_b9,r12 - 20668026.8748422 GlobalBidder_hat{W}__i9,k0
        >= -20668026.8748422
 c404:  aux_cap_helper__b0,r0,band_UNPAIRED__condX2
        + aux_cap_helper__b0,r0,band_UNPAIRED__condX1 - X__b0,r0,band_UNPAIRED
         = 0
 c405:  sv_function_helpervar_b4,r0__Z3 + sv_function_helpervar_b4,r0__Z1
        + sv_function_helpervar_b4,r0__Z2  = 1
 c406:  aux_cap_helper__b5,r0,band_LOW_PAIRED__condX1
        + aux_cap_helper__b5,r0,band_LOW_PAIRED__condX2
        - X__b5,r0,band_LOW_PAIRED  = 0
 c407:  sv_function_helpervar_b4,r4__condX3
        - 15.8453086887446 sv_function_helpervar_b4,r4__Z3 <= 0
 c408:  - 15.8453086887446 sv_function_helpervar_b1,r0__Z3
        + sv_function_helpervar_b1,r0__condX3 <= 0
 c409:  - aux_cap_helper__b0,r1,band_LOW_PAIRED__condX1 <= 0
 c410:  - aux_cap_helper__b0,r7,band_HIGH_PAIRED__condX1 <= 0
 c411:  aux_cap_helper__b4,r6,band_UNPAIRED__Z2
        + aux_cap_helper__b4,r6,band_UNPAIRED__Z1  = 1
 c412:  aux_cap_helper__b5,r4,band_UNPAIRED__condX1
        - aux_cap_helper__b5,r4,band_UNPAIRED__Z1 <= 0
 c413:  - sv_function_helpervar_b7,r7__condX2
        + 0.158453086898351 sv_function_helpervar_b7,r7__Z2 <= 0
 c414:  - X__b2,r5,band_LOW_PAIRED
        + aux_cap_helper__b2,r5,band_LOW_PAIRED__condX1
        + aux_cap_helper__b2,r5,band_LOW_PAIRED__condX2  = 0
 c415:  X__b8,r10,band_HIGH_PAIRED + X__b8,r10,band_LOW_PAIRED
        - GlobalBidder_W_ir__b8,r10 + X__b8,r10,band_UNPAIRED >= 0
 c416:  5.40314922344626 sv_function_helpervar_b3,r0__Z3
        - sv_function_helpervar_b3,r0__condX3 <= 0
 c417:  - 2 aux_cap_helper__b5,r1,band_LOW_PAIRED__Z2
        + aux_cap_helper__b5,r1,band_LOW_PAIRED__condX2 <= 0
 c418:  1496.02031229248 aux_sv_b2,r12 - aux_Omega_b2,r12  = 0
 c419:  - 2 aux_cap_helper__b7,r2,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b7,r2,band_HIGH_PAIRED__condX2 <= 0
 c420:  aux_cap_helper__b5,r4,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b5,r4,band_HIGH_PAIRED__condX2 <= 0
 c421:  - 15.8453086887446 sv_function_helpervar_b5,r11__Z3
        + sv_function_helpervar_b5,r11__condX3 <= 0
 c422:  4.4504289328 aux_cap_helper__b0,r11,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b0,r11,band_LOW_PAIRED__condX1
        - 1.2715511236 aux_cap_helper__b0,r11,band_LOW_PAIRED__Z2
        - aux_cap_b0,r11,band_LOW_PAIRED  = 0
 c423:  - aux_cap_helper__b4,r0,band_UNPAIRED__condX2
        + aux_cap_helper__b4,r0,band_UNPAIRED__Z2 <= 0
 c424:  aux_cap_helper__b8,r13,band_LOW_PAIRED__Z2
        + aux_cap_helper__b8,r13,band_LOW_PAIRED__Z1  = 1
 c425:  - aux_cap_helper__b0,r6,band_HIGH_PAIRED__condX1 <= 0
 c426:  - aux_cap_b2,r2,band_HIGH_PAIRED
        + 2.8394335959 aux_cap_helper__b2,r2,band_HIGH_PAIRED__condX2
        + 1.8253501688 aux_cap_helper__b2,r2,band_HIGH_PAIRED__condX3
        + 2.0281668543 aux_cap_helper__b2,r2,band_HIGH_PAIRED__condX1
        + 1.21690011250031 aux_cap_helper__b2,r2,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b2,r2,band_HIGH_PAIRED__Z2  = 0
 c427:  2 aux_cap_helper__b5,r4,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b5,r4,band_HIGH_PAIRED__condX3 <= 0
 c428:  - aux_cap_helper__b0,r0,band_HIGH_PAIRED__condX1 <= 0
 c429:  - X__b2,r12,band_HIGH_PAIRED
        + aux_cap_helper__b2,r12,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b2,r12,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b2,r12,band_HIGH_PAIRED__condX3  = 0
 c430:  - aux_cap_helper__b7,r8,band_HIGH_PAIRED__condX1 <= 0
 c431:  - sv_function_helpervar_b3,r5__condX1 <= 0
 c432:  aux_cap_helper__b3,r3,band_UNPAIRED__condX2
        + aux_cap_helper__b3,r3,band_UNPAIRED__condX1 - X__b3,r3,band_UNPAIRED
         = 0
 c433:  - aux_cap_b8,r13,band_UNPAIRED
        + 0.63460472 aux_cap_helper__b8,r13,band_UNPAIRED__condX1
        + 0.888446608 aux_cap_helper__b8,r13,band_UNPAIRED__condX2
        - 0.253841888 aux_cap_helper__b8,r13,band_UNPAIRED__Z2  = 0
 c434:  aux_cap_helper__b0,r10,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b0,r10,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b0,r10,band_HIGH_PAIRED__Z3  = 1
 c435:  - 0.253841888 aux_cap_helper__b9,r1,band_UNPAIRED__Z2
        - aux_cap_b9,r1,band_UNPAIRED
        + 0.63460472 aux_cap_helper__b9,r1,band_UNPAIRED__condX1
        + 0.888446608 aux_cap_helper__b9,r1,band_UNPAIRED__condX2  = 0
 c436:  - aux_cap_helper__b0,r6,band_UNPAIRED__condX1 <= 0
 c437:  aux_cap_helper__b3,r12,band_UNPAIRED__Z1
        + aux_cap_helper__b3,r12,band_UNPAIRED__Z2  = 1
 c438:  - aux_cap_helper__b4,r5,band_LOW_PAIRED__condX2
        + aux_cap_helper__b4,r5,band_LOW_PAIRED__Z2 <= 0
 c439:  0.158453086934078 sv_function_helpervar_b5,r2__Z2
        - sv_function_helpervar_b5,r2__condX2 <= 0
 c440:  - aux_cap_helper__b7,r6,band_LOW_PAIRED__condX1 <= 0
 c441:  - aux_c_b7,r9 + aux_cap_b7,r9,band_HIGH_PAIRED
        + aux_cap_b7,r9,band_LOW_PAIRED + aux_cap_b7,r9,band_UNPAIRED  = 0
 c442:  2 aux_cap_helper__b8,r5,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b8,r5,band_HIGH_PAIRED__condX3 <= 0
 c443:  - aux_cap_helper__b2,r2,band_UNPAIRED__condX2
        + aux_cap_helper__b2,r2,band_UNPAIRED__Z2 <= 0
 c444:  aux_cap_helper__b5,r3,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b5,r3,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b5,r3,band_HIGH_PAIRED__Z1  = 1
 c445:  aux_cap_helper__b1,r0,band_UNPAIRED__Z2
        + aux_cap_helper__b1,r0,band_UNPAIRED__Z1  = 1
 c446:  X__b0,r7,band_UNPAIRED + X__b2,r7,band_UNPAIRED
        + X__b1,r7,band_UNPAIRED + X__b7,r7,band_UNPAIRED
        + X__b5,r7,band_UNPAIRED + X__b6,r7,band_UNPAIRED
        + X__b3,r7,band_UNPAIRED + X__b8,r7,band_UNPAIRED
        + X__b9,r7,band_UNPAIRED + X__b4,r7,band_UNPAIRED <= 2
 c447:  - aux_cap_helper__b5,r1,band_LOW_PAIRED__Z1
        + aux_cap_helper__b5,r1,band_LOW_PAIRED__condX1 <= 0
 c448:  - 6.6034700582062 sv_function_helpervar_b8,r5__Z2
        + sv_function_helpervar_b8,r5__condX2 <= 0
 c449:  2.0281668543 aux_cap_helper__b6,r0,band_HIGH_PAIRED__condX1
        + 1.8253501688 aux_cap_helper__b6,r0,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b6,r0,band_HIGH_PAIRED__condX2
        - aux_cap_b6,r0,band_HIGH_PAIRED
        - 0.8112667416 aux_cap_helper__b6,r0,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b6,r0,band_HIGH_PAIRED__Z3  = 0
 c450:  - aux_c_b7,r1 + sv_function_helpervar_b7,r1__condX2
        + sv_function_helpervar_b7,r1__condX1
        + sv_function_helpervar_b7,r1__condX3  = 0
 c451:  aux_cap_helper__b5,r7,band_LOW_PAIRED__condX1
        + aux_cap_helper__b5,r7,band_LOW_PAIRED__condX2
        - X__b5,r7,band_LOW_PAIRED  = 0
 c452:  aux_cap_helper__b5,r9,band_LOW_PAIRED__Z2
        + aux_cap_helper__b5,r9,band_LOW_PAIRED__Z1  = 1
 c453:  - aux_cap_b6,r6,band_UNPAIRED
        + 0.63460472 aux_cap_helper__b6,r6,band_UNPAIRED__condX1
        + 0.888446608 aux_cap_helper__b6,r6,band_UNPAIRED__condX2
        - 0.253841888 aux_cap_helper__b6,r6,band_UNPAIRED__Z2  = 0
 c454:  - sv_function_helpervar_b0,r5__condX3
        + 6.46449030410519 sv_function_helpervar_b0,r5__Z3 <= 0
 c455:  - 2 aux_cap_helper__b4,r11,band_UNPAIRED__Z2
        + aux_cap_helper__b4,r11,band_UNPAIRED__condX2 <= 0
 c456:  - aux_cap_helper__b8,r2,band_UNPAIRED__Z1
        + aux_cap_helper__b8,r2,band_UNPAIRED__condX1 <= 0
 c457:  - aux_cap_helper__b6,r0,band_HIGH_PAIRED__condX1 <= 0
 c458:  aux_cap_helper__b1,r6,band_UNPAIRED__condX2
        + aux_cap_helper__b1,r6,band_UNPAIRED__condX1 - X__b1,r6,band_UNPAIRED
         = 0
 c459:  sv_function_helpervar_b5,r10__Z1 + sv_function_helpervar_b5,r10__Z2
        + sv_function_helpervar_b5,r10__Z3  = 1
 c460:  aux_cap_helper__b0,r10,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b0,r10,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b0,r10,band_HIGH_PAIRED__condX3
        - X__b0,r10,band_HIGH_PAIRED  = 0
 c461:  sv_function_helpervar_b0,r6__Z3 + sv_function_helpervar_b0,r6__Z1
        + sv_function_helpervar_b0,r6__Z2  = 1
 c462:  aux_cap_helper__b3,r4,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b3,r4,band_UNPAIRED__Z2 <= 0
 c463:  - aux_cap_b4,r2,band_LOW_PAIRED
        - 1.2715511236 aux_cap_helper__b4,r2,band_LOW_PAIRED__Z2
        + 4.4504289328 aux_cap_helper__b4,r2,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b4,r2,band_LOW_PAIRED__condX1  = 0
 c464:  aux_cap_helper__b2,r6,band_LOW_PAIRED__Z2
        + aux_cap_helper__b2,r6,band_LOW_PAIRED__Z1  = 1
 c465:  - sv_function_helpervar_b7,r10__condX1 <= 0
 c466:  - 2 aux_cap_helper__b8,r2,band_UNPAIRED__Z2
        + aux_cap_helper__b8,r2,band_UNPAIRED__condX2 <= 0
 c467:  - sv_function_helpervar_b1,r9__condX2
        + 0.158453086895131 sv_function_helpervar_b1,r9__Z2 <= 0
 c468:  aux_cap_helper__b3,r11,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b3,r11,band_HIGH_PAIRED__Z3 <= 0
 c469:  aux_cap_helper__b8,r3,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b8,r3,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b8,r3,band_HIGH_PAIRED__Z2  = 1
 c470:  sv_function_helpervar_b3,r4__Z1 + sv_function_helpervar_b3,r4__Z3
        + sv_function_helpervar_b3,r4__Z2  = 1
 c471:  GlobalBidder_W_i_7 + 14 GlobalBidder_hat{W}__i7,k0 <= 28
 c472:  aux_cap_helper__b9,r11,band_UNPAIRED__Z2
        + aux_cap_helper__b9,r11,band_UNPAIRED__Z1  = 1
 c473:  - 1.2715511236 aux_cap_helper__b2,r1,band_LOW_PAIRED__Z2
        - aux_cap_b2,r1,band_LOW_PAIRED
        + 4.4504289328 aux_cap_helper__b2,r1,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b2,r1,band_LOW_PAIRED__condX1  = 0
 c474:  2.0281668543 aux_cap_helper__b1,r11,band_HIGH_PAIRED__condX1
        + 2.8394335959 aux_cap_helper__b1,r11,band_HIGH_PAIRED__condX2
        + 1.8253501688 aux_cap_helper__b1,r11,band_HIGH_PAIRED__condX3
        + 1.21690011250031 aux_cap_helper__b1,r11,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b1,r11,band_HIGH_PAIRED__Z2
        - aux_cap_b1,r11,band_HIGH_PAIRED  = 0
 c475:  aux_cap_helper__b4,r11,band_UNPAIRED__condX2
        + aux_cap_helper__b4,r11,band_UNPAIRED__condX1
        - X__b4,r11,band_UNPAIRED  = 0
 c476:  - aux_cap_b6,r12,band_HIGH_PAIRED
        + 2.8394335959 aux_cap_helper__b6,r12,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b6,r12,band_HIGH_PAIRED__condX1
        + 1.8253501688 aux_cap_helper__b6,r12,band_HIGH_PAIRED__condX3
        + 1.21690011250031 aux_cap_helper__b6,r12,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b6,r12,band_HIGH_PAIRED__Z2  = 0
 c477:  sv_function_helpervar_b7,r11__condX3
        - 15.8453086887446 sv_function_helpervar_b7,r11__Z3 <= 0
 c478:  - aux_cap_helper__b9,r4,band_UNPAIRED__condX1 <= 0
 c479:  - aux_sv_b0,r8 + 53.0367824750787 sv_function_helpervar_b0,r8__Z3
        + 24.8045601418468 sv_function_helpervar_b0,r8__Z2
        + 163.7425743659 sv_function_helpervar_b0,r8__condX1
        + 7.2005932481 sv_function_helpervar_b0,r8__condX2
        + 2.7173798532 sv_function_helpervar_b0,r8__condX3  = 0
 c480:  - aux_c_b2,r12 + sv_function_helpervar_b2,r12__condX2
        + sv_function_helpervar_b2,r12__condX3
        + sv_function_helpervar_b2,r12__condX1  = 0
 c481:  - aux_cap_helper__b4,r11,band_UNPAIRED__Z1
        + aux_cap_helper__b4,r11,band_UNPAIRED__condX1 <= 0
 c482:  - sv_function_helpervar_b9,r10__condX3
        + 6.87475654633498 sv_function_helpervar_b9,r10__Z3 <= 0
 c483:  - 2 aux_cap_helper__b1,r10,band_LOW_PAIRED__Z2
        + aux_cap_helper__b1,r10,band_LOW_PAIRED__condX2 <= 0
 c484:  aux_cap_helper__b4,r10,band_UNPAIRED__Z2
        + aux_cap_helper__b4,r10,band_UNPAIRED__Z1  = 1
 c485:  - aux_cap_helper__b7,r2,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b7,r2,band_HIGH_PAIRED__condX1 <= 0
 c486:  - aux_cap_helper__b6,r12,band_UNPAIRED__condX1 <= 0
 c487:  - aux_cap_helper__b7,r10,band_LOW_PAIRED__condX1 <= 0
 c488:  - X__b8,r10,band_LOW_PAIRED
        + aux_cap_helper__b8,r10,band_LOW_PAIRED__condX1
        + aux_cap_helper__b8,r10,band_LOW_PAIRED__condX2  = 0
 c489:  1224.02722726178 aux_sv_b5,r11 - aux_Omega_b5,r11  = 0
 c490:  sv_function_helpervar_b9,r0__condX2
        - 6.29021169207404 sv_function_helpervar_b9,r0__Z2 <= 0
 c491:  aux_cap_helper__b4,r8,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b4,r8,band_HIGH_PAIRED__Z1 <= 0
 c492:  sv_function_helpervar_b4,r8__condX1
        - 0.158453086912977 sv_function_helpervar_b4,r8__Z1 <= 0
 c493:  aux_cap_helper__b3,r5,band_LOW_PAIRED__Z2
        - aux_cap_helper__b3,r5,band_LOW_PAIRED__condX2 <= 0
 c494:  - aux_c_b6,r5 + sv_function_helpervar_b6,r5__condX1
        + sv_function_helpervar_b6,r5__condX3
        + sv_function_helpervar_b6,r5__condX2  = 0
 c495:  - X__b7,r7,band_LOW_PAIRED
        + aux_cap_helper__b7,r7,band_LOW_PAIRED__condX2
        + aux_cap_helper__b7,r7,band_LOW_PAIRED__condX1  = 0
 c496:  - 3 aux_cap_helper__b5,r8,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b5,r8,band_HIGH_PAIRED__condX3 <= 0
 c497:  aux_cap_helper__b4,r8,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b4,r8,band_HIGH_PAIRED__Z2 <= 0
 c498:  - sv_function_helpervar_b6,r1__condX1 <= 0
 c499:  aux_cap_helper__b3,r4,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b3,r4,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b3,r4,band_HIGH_PAIRED__Z3  = 1
 c500:  - 15.8453086887446 sv_function_helpervar_b7,r9__Z3
        + sv_function_helpervar_b7,r9__condX3 <= 0
 c501:  - aux_cap_helper__b1,r1,band_LOW_PAIRED__condX1 <= 0
 c502:  - aux_sv_b8,r7 + 1636.3575432272 sv_function_helpervar_b8,r7__condX1
        + 28.7861276029 sv_function_helpervar_b8,r7__condX3
        + 66.134305782 sv_function_helpervar_b8,r7__condX2
        + 504.193085636999 sv_function_helpervar_b8,r7__Z3
        + 248.806719226985 sv_function_helpervar_b8,r7__Z2  = 0
 c503:  - aux_cap_b3,r11,band_UNPAIRED
        - 0.253841888 aux_cap_helper__b3,r11,band_UNPAIRED__Z2
        + 0.63460472 aux_cap_helper__b3,r11,band_UNPAIRED__condX1
        + 0.888446608 aux_cap_helper__b3,r11,band_UNPAIRED__condX2  = 0
 c504:  942.4308564564 sv_function_helpervar_b6,r2__condX1
        + 43.2858200351 sv_function_helpervar_b6,r2__condX2
        + 15.223457777 sv_function_helpervar_b6,r2__condX3
        + 311.857680471445 sv_function_helpervar_b6,r2__Z3
        + 142.47230662419 sv_function_helpervar_b6,r2__Z2 - aux_sv_b6,r2  = 0
 c505:  - aux_cap_helper__b6,r1,band_HIGH_PAIRED__condX1 <= 0
 c506:  sv_function_helpervar_b1,r0__condX2
        + sv_function_helpervar_b1,r0__condX3
        + sv_function_helpervar_b1,r0__condX1 - aux_c_b1,r0  = 0
 c507:  aux_cap_helper__b2,r6,band_UNPAIRED__condX2
        + aux_cap_helper__b2,r6,band_UNPAIRED__condX1 - X__b2,r6,band_UNPAIRED
         = 0
 c508:  470.06563575012 aux_sv_b0,r3 - aux_Omega_b0,r3  = 0
 c509:  2.8394335959 aux_cap_helper__b6,r4,band_HIGH_PAIRED__condX2
        + 1.8253501688 aux_cap_helper__b6,r4,band_HIGH_PAIRED__condX3
        + 2.0281668543 aux_cap_helper__b6,r4,band_HIGH_PAIRED__condX1
        + 1.21690011250031 aux_cap_helper__b6,r4,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b6,r4,band_HIGH_PAIRED__Z2
        - aux_cap_b6,r4,band_HIGH_PAIRED  = 0
 c510:  - 0.158453086899434 sv_function_helpervar_b3,r6__Z1
        + sv_function_helpervar_b3,r6__condX1 <= 0
 c511:  - aux_cap_b1,r9,band_UNPAIRED
        - 0.253841888 aux_cap_helper__b1,r9,band_UNPAIRED__Z2
        + 0.888446608 aux_cap_helper__b1,r9,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b1,r9,band_UNPAIRED__condX1  = 0
 c512:  - sv_function_helpervar_b2,r4__condX3
        + 6.2359236486132 sv_function_helpervar_b2,r4__Z3 <= 0
 c513:  - aux_Omega_b9,r7 + 4842.64208786644 aux_sv_b9,r7  = 0
 c514:  - aux_cap_helper__b6,r9,band_LOW_PAIRED__condX1 <= 0
 c515:  0.0102040816326531 X__b8,r6,band_HIGH_PAIRED
        + 0.0102040816326531 X__b8,r6,band_UNPAIRED - GlobalBidder_W_ir__b8,r6
        + 0.0102040816326531 X__b8,r6,band_LOW_PAIRED <= 0
 c516:  - 15.8453086887446 sv_function_helpervar_b0,r1__Z3
        + sv_function_helpervar_b0,r1__condX3 <= 0
 c517:  aux_cap_helper__b6,r1,band_LOW_PAIRED__Z2
        - aux_cap_helper__b6,r1,band_LOW_PAIRED__condX2 <= 0
 c518:  2 aux_cap_helper__b5,r5,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b5,r5,band_HIGH_PAIRED__condX3 <= 0
 c519:  0.888446608 aux_cap_helper__b3,r8,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b3,r8,band_UNPAIRED__condX1
        - 0.253841888 aux_cap_helper__b3,r8,band_UNPAIRED__Z2
        - aux_cap_b3,r8,band_UNPAIRED  = 0
 c520:  aux_cap_helper__b1,r9,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b1,r9,band_HIGH_PAIRED__Z2 <= 0
 c521:  aux_cap_helper__b0,r1,band_UNPAIRED__Z1
        + aux_cap_helper__b0,r1,band_UNPAIRED__Z2  = 1
 c522:  aux_cap_helper__b8,r1,band_UNPAIRED__condX1
        + aux_cap_helper__b8,r1,band_UNPAIRED__condX2 - X__b8,r1,band_UNPAIRED
         = 0
 c523:  - aux_cap_helper__b2,r5,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b2,r5,band_HIGH_PAIRED__Z3 <= 0
 c524:  - 0.8112667416 aux_cap_helper__b0,r9,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b0,r9,band_HIGH_PAIRED__Z3
        - aux_cap_b0,r9,band_HIGH_PAIRED
        + 1.8253501688 aux_cap_helper__b0,r9,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b0,r9,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b0,r9,band_HIGH_PAIRED__condX1  = 0
 c525:  0.0102040816326531 X__b7,r8,band_UNPAIRED - GlobalBidder_W_ir__b7,r8
        + 0.0102040816326531 X__b7,r8,band_HIGH_PAIRED
        + 0.0102040816326531 X__b7,r8,band_LOW_PAIRED <= 0
 c526:  aux_cap_helper__b0,r13,band_UNPAIRED__Z2
        - aux_cap_helper__b0,r13,band_UNPAIRED__condX2 <= 0
 c527:  sv_function_helpervar_b7,r12__condX2
        - 6.80514243953476 sv_function_helpervar_b7,r12__Z2 <= 0
 c528:  44.3472714559725 aux_sv_b0,r6 - aux_Omega_b0,r6  = 0
 c529:  2 aux_cap_helper__b8,r4,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b8,r4,band_HIGH_PAIRED__condX3 <= 0
 c530:  aux_cap_helper__b8,r0,band_LOW_PAIRED__condX1
        - aux_cap_helper__b8,r0,band_LOW_PAIRED__Z1 <= 0
 c531:  aux_cap_helper__b6,r3,band_UNPAIRED__Z2
        + aux_cap_helper__b6,r3,band_UNPAIRED__Z1  = 1
 c532:  aux_cap_helper__b0,r4,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b0,r4,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b0,r4,band_HIGH_PAIRED__Z1  = 1
 c533:  aux_cap_helper__b1,r5,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b1,r5,band_HIGH_PAIRED__Z3 <= 0
 c534:  aux_cap_helper__b1,r11,band_UNPAIRED__condX1
        + aux_cap_helper__b1,r11,band_UNPAIRED__condX2
        - X__b1,r11,band_UNPAIRED  = 0
 c535:  aux_cap_helper__b8,r5,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b8,r5,band_HIGH_PAIRED__condX2 <= 0
 c536:  - aux_cap_helper__b6,r6,band_UNPAIRED__condX1 <= 0
 c537:  aux_cap_helper__b8,r0,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b8,r0,band_LOW_PAIRED__Z2 <= 0
 c538:  - 3 aux_cap_helper__b8,r7,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b8,r7,band_HIGH_PAIRED__condX3 <= 0
 c539:  aux_cap_helper__b5,r5,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b5,r5,band_HIGH_PAIRED__condX2 <= 0
 c540:  aux_cap_helper__b2,r5,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b2,r5,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b2,r5,band_HIGH_PAIRED__Z2  = 1
 c541:  aux_cap_helper__b1,r9,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b1,r9,band_HIGH_PAIRED__Z1 <= 0
 c542:  aux_cap_helper__b2,r13,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b2,r13,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b2,r13,band_HIGH_PAIRED__Z1  = 1
 c543:  6.26935260585317 sv_function_helpervar_b2,r11__Z3
        - sv_function_helpervar_b2,r11__condX3 <= 0
 c544:  aux_cap_helper__b6,r6,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b6,r6,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b6,r6,band_HIGH_PAIRED__condX1
        - X__b6,r6,band_HIGH_PAIRED  = 0
 c545:  - aux_cap_helper__b2,r5,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b2,r5,band_HIGH_PAIRED__Z2 <= 0
 c546:  aux_cap_helper__b7,r11,band_UNPAIRED__Z2
        - aux_cap_helper__b7,r11,band_UNPAIRED__condX2 <= 0
 c547:  - aux_cap_helper__b1,r9,band_LOW_PAIRED__condX2
        + aux_cap_helper__b1,r9,band_LOW_PAIRED__Z2 <= 0
 c548:  aux_cap_helper__b3,r13,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b3,r13,band_HIGH_PAIRED__Z2 <= 0
 c549:  - 3 aux_cap_helper__b1,r0,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b1,r0,band_HIGH_PAIRED__condX3 <= 0
 c550:  2.8394335959 aux_cap_helper__b4,r8,band_HIGH_PAIRED__condX2
        + 1.8253501688 aux_cap_helper__b4,r8,band_HIGH_PAIRED__condX3
        + 2.0281668543 aux_cap_helper__b4,r8,band_HIGH_PAIRED__condX1
        + 1.21690011250031 aux_cap_helper__b4,r8,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b4,r8,band_HIGH_PAIRED__Z2
        - aux_cap_b4,r8,band_HIGH_PAIRED  = 0
 c551:  aux_cap_helper__b4,r2,band_LOW_PAIRED__Z2
        - aux_cap_helper__b4,r2,band_LOW_PAIRED__condX2 <= 0
 c552:  - aux_cap_b8,r1,band_HIGH_PAIRED
        + 2.0281668543 aux_cap_helper__b8,r1,band_HIGH_PAIRED__condX1
        + 2.8394335959 aux_cap_helper__b8,r1,band_HIGH_PAIRED__condX2
        + 1.8253501688 aux_cap_helper__b8,r1,band_HIGH_PAIRED__condX3
        - 0.8112667416 aux_cap_helper__b8,r1,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b8,r1,band_HIGH_PAIRED__Z3  = 0
 c553:  - aux_Omega_b2,r0 + 106.425222405369 aux_sv_b2,r0  = 0
 c554:  2.0281668543 aux_cap_helper__b3,r13,band_HIGH_PAIRED__condX1
        + 2.8394335959 aux_cap_helper__b3,r13,band_HIGH_PAIRED__condX2
        + 1.8253501688 aux_cap_helper__b3,r13,band_HIGH_PAIRED__condX3
        - 0.8112667416 aux_cap_helper__b3,r13,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b3,r13,band_HIGH_PAIRED__Z3
        - aux_cap_b3,r13,band_HIGH_PAIRED  = 0
 c555:  aux_cap_helper__b1,r3,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b1,r3,band_HIGH_PAIRED__Z2 <= 0
 c556:  - sv_function_helpervar_b2,r13__condX1 <= 0
 c557:  - sv_function_helpervar_b0,r0__condX1 <= 0
 c558:  - 0.158453086994015 sv_function_helpervar_b2,r9__Z1
        + sv_function_helpervar_b2,r9__condX1 <= 0
 c559:  aux_cap_helper__b3,r0,band_LOW_PAIRED__Z1
        + aux_cap_helper__b3,r0,band_LOW_PAIRED__Z2  = 1
 c560:  aux_cap_helper__b7,r2,band_UNPAIRED__Z1
        + aux_cap_helper__b7,r2,band_UNPAIRED__Z2  = 1
 c561:  - 0.158453086896477 sv_function_helpervar_b8,r9__Z1
        + sv_function_helpervar_b8,r9__condX1 <= 0
 c562:  X__b5,r6,band_HIGH_PAIRED + X__b1,r6,band_HIGH_PAIRED
        + X__b8,r6,band_HIGH_PAIRED + X__b2,r6,band_HIGH_PAIRED
        + X__b4,r6,band_HIGH_PAIRED + X__b3,r6,band_HIGH_PAIRED
        + X__b7,r6,band_HIGH_PAIRED + X__b6,r6,band_HIGH_PAIRED
        + X__b0,r6,band_HIGH_PAIRED + X__b9,r6,band_HIGH_PAIRED <= 3
 c563:  sv_function_helpervar_b3,r10__condX3
        - 15.8453086887446 sv_function_helpervar_b3,r10__Z3 <= 0
 c564:  - aux_cap_helper__b1,r10,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b1,r10,band_HIGH_PAIRED__condX1 <= 0
 c565:  - aux_Omega_b5,r6 + 51.0428798694979 aux_sv_b5,r6  = 0
 c566:  aux_cap_helper__b9,r12,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b9,r12,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b9,r12,band_HIGH_PAIRED__Z1  = 1
 c567:  4.4504289328 aux_cap_helper__b1,r11,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b1,r11,band_LOW_PAIRED__condX1
        - 1.2715511236 aux_cap_helper__b1,r11,band_LOW_PAIRED__Z2
        - aux_cap_b1,r11,band_LOW_PAIRED  = 0
 c568:  aux_cap_b2,r2,band_HIGH_PAIRED - aux_c_b2,r2
        + aux_cap_b2,r2,band_LOW_PAIRED + aux_cap_b2,r2,band_UNPAIRED  = 0
 c569:  - X__b9,r7,band_LOW_PAIRED
        + aux_cap_helper__b9,r7,band_LOW_PAIRED__condX1
        + aux_cap_helper__b9,r7,band_LOW_PAIRED__condX2  = 0
 c570:  - aux_sv_b8,r13 + 249.75325043164 sv_function_helpervar_b8,r13__Z2
        + 467.937707603433 sv_function_helpervar_b8,r13__Z3
        + 1636.3575440595 sv_function_helpervar_b8,r13__condX1
        + 60.1607320396 sv_function_helpervar_b8,r13__condX2
        + 31.0742103879 sv_function_helpervar_b8,r13__condX3  = 0
 c571:  - aux_cap_helper__b0,r4,band_LOW_PAIRED__condX1 <= 0
 c572:  - 2 aux_cap_helper__b1,r10,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b1,r10,band_HIGH_PAIRED__condX2 <= 0
 c573:  sv_function_helpervar_b3,r8__condX2
        - 6.12592415881159 sv_function_helpervar_b3,r8__Z2 <= 0
 c574:  - aux_cap_helper__b6,r12,band_HIGH_PAIRED__condX1 <= 0
 c575:  - 1.2715511236 aux_cap_helper__b2,r11,band_LOW_PAIRED__Z2
        + 4.4504289328 aux_cap_helper__b2,r11,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b2,r11,band_LOW_PAIRED__condX1
        - aux_cap_b2,r11,band_LOW_PAIRED  = 0
 c576:  1.8253501688 aux_cap_helper__b4,r3,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b4,r3,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b4,r3,band_HIGH_PAIRED__condX1
        - aux_cap_b4,r3,band_HIGH_PAIRED
        + 1.21690011250031 aux_cap_helper__b4,r3,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b4,r3,band_HIGH_PAIRED__Z2  = 0
 c577:  aux_cap_helper__b6,r3,band_LOW_PAIRED__Z1
        + aux_cap_helper__b6,r3,band_LOW_PAIRED__Z2  = 1
 c578:  - X__b7,r1,band_HIGH_PAIRED
        + aux_cap_helper__b7,r1,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b7,r1,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b7,r1,band_HIGH_PAIRED__condX1  = 0
 c579:  - aux_c_b3,r0 + aux_cap_b3,r0,band_HIGH_PAIRED
        + aux_cap_b3,r0,band_UNPAIRED + aux_cap_b3,r0,band_LOW_PAIRED  = 0
 c580:  - 1.2715511236 aux_cap_helper__b1,r13,band_LOW_PAIRED__Z2
        + 3.1788778092 aux_cap_helper__b1,r13,band_LOW_PAIRED__condX1
        + 4.4504289328 aux_cap_helper__b1,r13,band_LOW_PAIRED__condX2
        - aux_cap_b1,r13,band_LOW_PAIRED  = 0
 c581:  - sv_function_helpervar_b8,r7__condX1 <= 0
 c582:  - 15.8453086887446 sv_function_helpervar_b9,r12__Z3
        + sv_function_helpervar_b9,r12__condX3 <= 0
 c583:  - 2 aux_cap_helper__b5,r4,band_LOW_PAIRED__Z2
        + aux_cap_helper__b5,r4,band_LOW_PAIRED__condX2 <= 0
 c584:  - aux_cap_b2,r11,band_UNPAIRED
        + 0.888446608 aux_cap_helper__b2,r11,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b2,r11,band_UNPAIRED__condX1
        - 0.253841888 aux_cap_helper__b2,r11,band_UNPAIRED__Z2  = 0
 c585:  - aux_cap_helper__b3,r1,band_HIGH_PAIRED__condX1 <= 0
 c586:  - 15.8453086887446 sv_function_helpervar_b5,r3__Z3
        + sv_function_helpervar_b5,r3__condX3 <= 0
 c587:  X__b4,r7,band_LOW_PAIRED + X__b6,r7,band_LOW_PAIRED
        + X__b2,r7,band_LOW_PAIRED + X__b9,r7,band_LOW_PAIRED
        + X__b7,r7,band_LOW_PAIRED + X__b8,r7,band_LOW_PAIRED
        + X__b5,r7,band_LOW_PAIRED + X__b3,r7,band_LOW_PAIRED
        + X__b1,r7,band_LOW_PAIRED + X__b0,r7,band_LOW_PAIRED <= 2
 c588:  aux_cap_helper__b7,r2,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b7,r2,band_UNPAIRED__Z2 <= 0
 c589:  aux_cap_helper__b1,r3,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b1,r3,band_HIGH_PAIRED__Z1 <= 0
 c590:  - aux_cap_helper__b0,r3,band_UNPAIRED__Z1
        + aux_cap_helper__b0,r3,band_UNPAIRED__condX1 <= 0
 c591:  aux_cap_helper__b3,r10,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b3,r10,band_HIGH_PAIRED__Z3 <= 0
 c592:  0.63460472 aux_cap_helper__b1,r11,band_UNPAIRED__condX1
        + 0.888446608 aux_cap_helper__b1,r11,band_UNPAIRED__condX2
        - 0.253841888 aux_cap_helper__b1,r11,band_UNPAIRED__Z2
        - aux_cap_b1,r11,band_UNPAIRED  = 0
 c593:  1.21690011250031 aux_cap_helper__b8,r6,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b8,r6,band_HIGH_PAIRED__Z2
        - aux_cap_b8,r6,band_HIGH_PAIRED
        + 2.0281668543 aux_cap_helper__b8,r6,band_HIGH_PAIRED__condX1
        + 2.8394335959 aux_cap_helper__b8,r6,band_HIGH_PAIRED__condX2
        + 1.8253501688 aux_cap_helper__b8,r6,band_HIGH_PAIRED__condX3  = 0
 c594:  sv_function_helpervar_b0,r1__Z1 + sv_function_helpervar_b0,r1__Z3
        + sv_function_helpervar_b0,r1__Z2  = 1
 c595:  - 1.2715511236 aux_cap_helper__b9,r0,band_LOW_PAIRED__Z2
        - aux_cap_b9,r0,band_LOW_PAIRED
        + 3.1788778092 aux_cap_helper__b9,r0,band_LOW_PAIRED__condX1
        + 4.4504289328 aux_cap_helper__b9,r0,band_LOW_PAIRED__condX2  = 0
 c596:  - 5.94902634456456 sv_function_helpervar_b3,r12__Z2
        + sv_function_helpervar_b3,r12__condX2 <= 0
 c597:  5.74546828281525 sv_function_helpervar_b5,r3__Z3
        - sv_function_helpervar_b5,r3__condX3 <= 0
 c598:  - aux_cap_b9,r8,band_UNPAIRED
        + 0.888446608 aux_cap_helper__b9,r8,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b9,r8,band_UNPAIRED__condX1
        - 0.253841888 aux_cap_helper__b9,r8,band_UNPAIRED__Z2  = 0
 c599:  - aux_cap_helper__b0,r1,band_HIGH_PAIRED__condX1 <= 0
 c600:  - aux_cap_helper__b7,r3,band_LOW_PAIRED__condX1 <= 0
 c601:  aux_cap_helper__b5,r8,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b5,r8,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b5,r8,band_HIGH_PAIRED__condX1
        - X__b5,r8,band_HIGH_PAIRED  = 0
 c602:  aux_cap_helper__b6,r11,band_LOW_PAIRED__Z1
        + aux_cap_helper__b6,r11,band_LOW_PAIRED__Z2  = 1
 c603:  - aux_c_b5,r12 + sv_function_helpervar_b5,r12__condX3
        + sv_function_helpervar_b5,r12__condX2
        + sv_function_helpervar_b5,r12__condX1  = 0
 c604:  aux_cap_b2,r9,band_UNPAIRED - aux_c_b2,r9
        + aux_cap_b2,r9,band_LOW_PAIRED + aux_cap_b2,r9,band_HIGH_PAIRED  = 0
 c605:  - sv_function_helpervar_b3,r7__condX1 <= 0
 c606:  - sv_function_helpervar_b9,r11__condX1 <= 0
 c607:  aux_cap_helper__b9,r1,band_UNPAIRED__Z2
        - aux_cap_helper__b9,r1,band_UNPAIRED__condX2 <= 0
 c608:  - 2 aux_cap_helper__b2,r5,band_UNPAIRED__Z2
        + aux_cap_helper__b2,r5,band_UNPAIRED__condX2 <= 0
 c609:  aux_cap_helper__b3,r7,band_UNPAIRED__Z1
        + aux_cap_helper__b3,r7,band_UNPAIRED__Z2  = 1
 c610:  0.15845308691419 sv_function_helpervar_b4,r0__Z2
        - sv_function_helpervar_b4,r0__condX2 <= 0
 c611:  aux_cap_helper__b4,r4,band_UNPAIRED__condX1
        - aux_cap_helper__b4,r4,band_UNPAIRED__Z1 <= 0
 c612:  - aux_cap_helper__b5,r4,band_LOW_PAIRED__Z1
        + aux_cap_helper__b5,r4,band_LOW_PAIRED__condX1 <= 0
 c613:  aux_cap_helper__b7,r4,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b7,r4,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b7,r4,band_HIGH_PAIRED__Z3  = 1
 c614:  sv_function_helpervar_b0,r0__Z3 + sv_function_helpervar_b0,r0__Z2
        + sv_function_helpervar_b0,r0__Z1  = 1
 c615:  - aux_Omega_b9,r9 + 313.051475020066 aux_sv_b9,r9  = 0
 c616:  sv_function_helpervar_b3,r8__condX3
        - 15.8453086887446 sv_function_helpervar_b3,r8__Z3 <= 0
 c617:  1.9054143854 sv_function_helpervar_b2,r3__condX3
        + 5.0379297428 sv_function_helpervar_b2,r3__condX2
        + 114.7166939392 sv_function_helpervar_b2,r3__condX1 - aux_sv_b2,r3
        + 17.3789387546259 sv_function_helpervar_b2,r3__Z2
        + 37.131136713186 sv_function_helpervar_b2,r3__Z3  = 0
 c618:  - aux_cap_helper__b8,r5,band_UNPAIRED__condX1 <= 0
 c619:  - sv_function_helpervar_b3,r13__condX3
        + 5.48961824768186 sv_function_helpervar_b3,r13__Z3 <= 0
 c620:  - aux_Omega_b6,r7 + 2340.905499019 aux_sv_b6,r7  = 0
 c621:  - aux_sv_b4,r11 + 287.992779139163 sv_function_helpervar_b4,r11__Z3
        + 133.744293427487 sv_function_helpervar_b4,r11__Z2
        + 39.3404038601 sv_function_helpervar_b4,r11__condX2
        + 883.4028075125 sv_function_helpervar_b4,r11__condX1
        + 14.5433515336 sv_function_helpervar_b4,r11__condX3  = 0
 c622:  sv_function_helpervar_b3,r9__Z1 + sv_function_helpervar_b3,r9__Z2
        + sv_function_helpervar_b3,r9__Z3  = 1
 c623:  aux_cap_b2,r10,band_LOW_PAIRED + aux_cap_b2,r10,band_HIGH_PAIRED
        - aux_c_b2,r10 + aux_cap_b2,r10,band_UNPAIRED  = 0
 c624:  5.52659420777561 sv_function_helpervar_b4,r3__Z3
        - sv_function_helpervar_b4,r3__condX3 <= 0
 c625:  - aux_cap_b0,r11,band_UNPAIRED
        - 0.253841888 aux_cap_helper__b0,r11,band_UNPAIRED__Z2
        + 0.888446608 aux_cap_helper__b0,r11,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b0,r11,band_UNPAIRED__condX1  = 0
 c626:  - aux_cap_helper__b8,r9,band_LOW_PAIRED__Z1
        + aux_cap_helper__b8,r9,band_LOW_PAIRED__condX1 <= 0
 c627:  - sv_function_helpervar_b2,r1__condX2
        + 0.15845308691457 sv_function_helpervar_b2,r1__Z2 <= 0
 c628:  - aux_cap_helper__b7,r13,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b7,r13,band_HIGH_PAIRED__Z2 <= 0
 c629:  aux_cap_helper__b4,r13,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b4,r13,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b4,r13,band_HIGH_PAIRED__Z2  = 1
 c630:  - X__b1,r12,band_LOW_PAIRED
        + aux_cap_helper__b1,r12,band_LOW_PAIRED__condX1
        + aux_cap_helper__b1,r12,band_LOW_PAIRED__condX2  = 0
 c631:  - 15.8453086887446 sv_function_helpervar_b0,r0__Z3
        + sv_function_helpervar_b0,r0__condX3 <= 0
 c632:  aux_cap_helper__b5,r0,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b5,r0,band_HIGH_PAIRED__condX2 <= 0
 c633:  - sv_function_helpervar_b3,r3__condX2
        + 0.158453086906449 sv_function_helpervar_b3,r3__Z2 <= 0
 c634:  sv_function_helpervar_b5,r1__condX2
        + sv_function_helpervar_b5,r1__condX1
        + sv_function_helpervar_b5,r1__condX3 - aux_c_b5,r1  = 0
 c635:  aux_cap_helper__b8,r1,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b8,r1,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b8,r1,band_HIGH_PAIRED__Z3  = 1
 c636:  aux_cap_helper__b8,r9,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b8,r9,band_HIGH_PAIRED__condX2 <= 0
 c637:  sv_function_helpervar_b9,r3__condX2
        - 7.8286385936948 sv_function_helpervar_b9,r3__Z2 <= 0
 c638:  - 2 aux_cap_helper__b8,r9,band_LOW_PAIRED__Z2
        + aux_cap_helper__b8,r9,band_LOW_PAIRED__condX2 <= 0
 c639:  sv_function_helpervar_b5,r9__condX2
        - 6.04014953766023 sv_function_helpervar_b5,r9__Z2 <= 0
 c640:  - aux_cap_helper__b7,r13,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b7,r13,band_HIGH_PAIRED__Z3 <= 0
 c641:  2 aux_cap_helper__b5,r0,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b5,r0,band_HIGH_PAIRED__condX3 <= 0
 c642:  2 aux_cap_helper__b1,r1,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b1,r1,band_HIGH_PAIRED__condX3 <= 0
 c643:  aux_cap_helper__b3,r0,band_LOW_PAIRED__condX1
        + aux_cap_helper__b3,r0,band_LOW_PAIRED__condX2
        - X__b3,r0,band_LOW_PAIRED  = 0
 c644:  2 aux_cap_helper__b8,r9,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b8,r9,band_HIGH_PAIRED__condX3 <= 0
 c645:  - 0.158453086933477 sv_function_helpervar_b3,r4__Z1
        + sv_function_helpervar_b3,r4__condX1 <= 0
 c646:  0.888446608 aux_cap_helper__b1,r6,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b1,r6,band_UNPAIRED__condX1
        - aux_cap_b1,r6,band_UNPAIRED
        - 0.253841888 aux_cap_helper__b1,r6,band_UNPAIRED__Z2  = 0
 c647:  - aux_cap_helper__b7,r4,band_UNPAIRED__condX1 <= 0
 c648:  - aux_cap_helper__b2,r4,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b2,r4,band_HIGH_PAIRED__Z3 <= 0
 c649:  0.63460472 aux_cap_helper__b4,r8,band_UNPAIRED__condX1
        + 0.888446608 aux_cap_helper__b4,r8,band_UNPAIRED__condX2
        - aux_cap_b4,r8,band_UNPAIRED
        - 0.253841888 aux_cap_helper__b4,r8,band_UNPAIRED__Z2  = 0
 c650:  318.21641444586 sv_function_helpervar_b5,r7__Z3
        + 148.24124206721 sv_function_helpervar_b5,r7__Z2 - aux_sv_b5,r7
        + 978.9018727381 sv_function_helpervar_b5,r7__condX1
        + 43.3489913322 sv_function_helpervar_b5,r7__condX2
        + 16.1729354303 sv_function_helpervar_b5,r7__condX3  = 0
 c651:  - aux_Omega_b2,r6 + 49.5268707942277 aux_sv_b2,r6  = 0
 c652:  - aux_cap_helper__b8,r10,band_LOW_PAIRED__Z1
        + aux_cap_helper__b8,r10,band_LOW_PAIRED__condX1 <= 0
 c653:  7.60706438507882 sv_function_helpervar_b8,r9__Z3
        - sv_function_helpervar_b8,r9__condX3 <= 0
 c654:  aux_cap_helper__b7,r11,band_LOW_PAIRED__condX2
        + aux_cap_helper__b7,r11,band_LOW_PAIRED__condX1
        - X__b7,r11,band_LOW_PAIRED  = 0
 c655:  aux_cap_helper__b0,r10,band_UNPAIRED__condX2
        + aux_cap_helper__b0,r10,band_UNPAIRED__condX1
        - X__b0,r10,band_UNPAIRED  = 0
 c656:  aux_cap_helper__b6,r6,band_LOW_PAIRED__Z1
        + aux_cap_helper__b6,r6,band_LOW_PAIRED__Z2  = 1
 c657:  5.45154056009326 sv_function_helpervar_b4,r10__Z3
        - sv_function_helpervar_b4,r10__condX3 <= 0
 c658:  aux_cap_helper__b0,r6,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b0,r6,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b0,r6,band_HIGH_PAIRED__condX2
        - X__b0,r6,band_HIGH_PAIRED  = 0
 c659:  - aux_cap_helper__b2,r5,band_UNPAIRED__condX1 <= 0
 c660:  aux_cap_b4,r3,band_LOW_PAIRED + aux_cap_b4,r3,band_UNPAIRED
        + aux_cap_b4,r3,band_HIGH_PAIRED - aux_c_b4,r3  = 0
 c661:  - sv_function_helpervar_b0,r2__condX1 <= 0
 c662:  3.1788778092 aux_cap_helper__b1,r5,band_LOW_PAIRED__condX1
        + 4.4504289328 aux_cap_helper__b1,r5,band_LOW_PAIRED__condX2
        - aux_cap_b1,r5,band_LOW_PAIRED
        - 1.2715511236 aux_cap_helper__b1,r5,band_LOW_PAIRED__Z2  = 0
 c663:  sv_function_helpervar_b1,r11__condX1
        + sv_function_helpervar_b1,r11__condX2
        + sv_function_helpervar_b1,r11__condX3 - aux_c_b1,r11  = 0
 c664:  - 2 aux_cap_helper__b8,r10,band_LOW_PAIRED__Z2
        + aux_cap_helper__b8,r10,band_LOW_PAIRED__condX2 <= 0
 c665:  - aux_cap_helper__b1,r11,band_UNPAIRED__condX1 <= 0
 c666:  - X__b3,r2,band_HIGH_PAIRED
        + aux_cap_helper__b3,r2,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b3,r2,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b3,r2,band_HIGH_PAIRED__condX3  = 0
 c667:  - aux_cap_helper__b4,r10,band_UNPAIRED__Z1
        + aux_cap_helper__b4,r10,band_UNPAIRED__condX1 <= 0
 c668:  - sv_function_helpervar_b5,r1__condX2
        + 0.158453086936865 sv_function_helpervar_b5,r1__Z2 <= 0
 c669:  - aux_cap_b6,r2,band_UNPAIRED
        - 0.253841888 aux_cap_helper__b6,r2,band_UNPAIRED__Z2
        + 0.63460472 aux_cap_helper__b6,r2,band_UNPAIRED__condX1
        + 0.888446608 aux_cap_helper__b6,r2,band_UNPAIRED__condX2  = 0
 c670:  - 0.253841888 aux_cap_helper__b9,r4,band_UNPAIRED__Z2
        - aux_cap_b9,r4,band_UNPAIRED
        + 0.63460472 aux_cap_helper__b9,r4,band_UNPAIRED__condX1
        + 0.888446608 aux_cap_helper__b9,r4,band_UNPAIRED__condX2  = 0
 c671:  0.63460472 aux_cap_helper__b3,r0,band_UNPAIRED__condX1
        + 0.888446608 aux_cap_helper__b3,r0,band_UNPAIRED__condX2
        - aux_cap_b3,r0,band_UNPAIRED
        - 0.253841888 aux_cap_helper__b3,r0,band_UNPAIRED__Z2  = 0
 c672:  aux_cap_helper__b4,r3,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b4,r3,band_HIGH_PAIRED__Z3 <= 0
 c673:  aux_cap_helper__b1,r1,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b1,r1,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b1,r1,band_HIGH_PAIRED__Z1  = 1
 c674:  - aux_cap_helper__b4,r9,band_HIGH_PAIRED__condX1 <= 0
 c675:  - X__b2,r1,band_UNPAIRED + aux_cap_helper__b2,r1,band_UNPAIRED__condX1
        + aux_cap_helper__b2,r1,band_UNPAIRED__condX2  = 0
 c676:  - aux_cap_helper__b0,r13,band_LOW_PAIRED__condX1 <= 0
 c677:  aux_cap_helper__b1,r1,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b1,r1,band_HIGH_PAIRED__condX2 <= 0
 c678:  aux_cap_helper__b4,r10,band_LOW_PAIRED__Z2
        + aux_cap_helper__b4,r10,band_LOW_PAIRED__Z1  = 1
 c679:  aux_cap_helper__b0,r3,band_UNPAIRED__Z2
        - aux_cap_helper__b0,r3,band_UNPAIRED__condX2 <= 0
 c680:  - sv_function_helpervar_b1,r11__condX3
        + 6.15161959363656 sv_function_helpervar_b1,r11__Z3 <= 0
 c681:  aux_cap_helper__b6,r4,band_LOW_PAIRED__Z2
        - aux_cap_helper__b6,r4,band_LOW_PAIRED__condX2 <= 0
 c682:  - aux_cap_b8,r11,band_HIGH_PAIRED
        + 1.21690011250031 aux_cap_helper__b8,r11,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b8,r11,band_HIGH_PAIRED__Z2
        + 2.0281668543 aux_cap_helper__b8,r11,band_HIGH_PAIRED__condX1
        + 1.8253501688 aux_cap_helper__b8,r11,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b8,r11,band_HIGH_PAIRED__condX2  = 0
 c683:  - aux_cap_b8,r4,band_LOW_PAIRED
        - 1.2715511236 aux_cap_helper__b8,r4,band_LOW_PAIRED__Z2
        + 3.1788778092 aux_cap_helper__b8,r4,band_LOW_PAIRED__condX1
        + 4.4504289328 aux_cap_helper__b8,r4,band_LOW_PAIRED__condX2  = 0
 c684:  aux_cap_helper__b1,r13,band_UNPAIRED__condX2
        + aux_cap_helper__b1,r13,band_UNPAIRED__condX1
        - X__b1,r13,band_UNPAIRED  = 0
 c685:  - aux_cap_helper__b7,r6,band_HIGH_PAIRED__condX1 <= 0
 c686:  - aux_cap_helper__b4,r2,band_UNPAIRED__condX2
        + aux_cap_helper__b4,r2,band_UNPAIRED__Z2 <= 0
 c687:  - aux_c_b6,r10 + sv_function_helpervar_b6,r10__condX3
        + sv_function_helpervar_b6,r10__condX2
        + sv_function_helpervar_b6,r10__condX1  = 0
 c688:  - X__b1,r1,band_HIGH_PAIRED
        + aux_cap_helper__b1,r1,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b1,r1,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b1,r1,band_HIGH_PAIRED__condX2  = 0
 c689:  - aux_c_b7,r6 + aux_cap_b7,r6,band_HIGH_PAIRED
        + aux_cap_b7,r6,band_UNPAIRED + aux_cap_b7,r6,band_LOW_PAIRED  = 0
 c690:  - aux_cap_b3,r9,band_LOW_PAIRED
        - 1.2715511236 aux_cap_helper__b3,r9,band_LOW_PAIRED__Z2
        + 4.4504289328 aux_cap_helper__b3,r9,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b3,r9,band_LOW_PAIRED__condX1  = 0
 c691:  - aux_Omega_b5,r4 + 819.857955143073 aux_sv_b5,r4  = 0
 c692:  aux_cap_helper__b6,r12,band_LOW_PAIRED__Z2
        + aux_cap_helper__b6,r12,band_LOW_PAIRED__Z1  = 1
 c693:  sv_function_helpervar_b9,r2__Z2 + sv_function_helpervar_b9,r2__Z3
        + sv_function_helpervar_b9,r2__Z1  = 1
 c694:  aux_cap_helper__b5,r4,band_UNPAIRED__Z1
        + aux_cap_helper__b5,r4,band_UNPAIRED__Z2  = 1
 c695:  - aux_Omega_b5,r2 + 139.597951212314 aux_sv_b5,r2  = 0
 c696:  sv_function_helpervar_b8,r4__condX1
        - 0.158453086919509 sv_function_helpervar_b8,r4__Z1 <= 0
 c697:  - aux_c_b1,r3 + sv_function_helpervar_b1,r3__condX1
        + sv_function_helpervar_b1,r3__condX2
        + sv_function_helpervar_b1,r3__condX3  = 0
 c698:  aux_cap_helper__b4,r1,band_LOW_PAIRED__condX2
        + aux_cap_helper__b4,r1,band_LOW_PAIRED__condX1
        - X__b4,r1,band_LOW_PAIRED  = 0
 c699:  - 3 aux_cap_helper__b8,r11,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b8,r11,band_HIGH_PAIRED__condX3 <= 0
 c700:  aux_cap_helper__b1,r8,band_LOW_PAIRED__Z2
        + aux_cap_helper__b1,r8,band_LOW_PAIRED__Z1  = 1
 c701:  - 1.2715511236 aux_cap_helper__b0,r5,band_LOW_PAIRED__Z2
        + 3.1788778092 aux_cap_helper__b0,r5,band_LOW_PAIRED__condX1
        + 4.4504289328 aux_cap_helper__b0,r5,band_LOW_PAIRED__condX2
        - aux_cap_b0,r5,band_LOW_PAIRED  = 0
 c702:  aux_cap_helper__b4,r8,band_UNPAIRED__Z1
        + aux_cap_helper__b4,r8,band_UNPAIRED__Z2  = 1
 c703:  - 2 aux_cap_helper__b9,r12,band_UNPAIRED__Z2
        + aux_cap_helper__b9,r12,band_UNPAIRED__condX2 <= 0
 c704:  17.4063585110541 sv_function_helpervar_b2,r9__Z2
        + 36.4228940769303 sv_function_helpervar_b2,r9__Z3 - aux_sv_b2,r9
        + 114.7166938733 sv_function_helpervar_b2,r9__condX1
        + 4.8648832127 sv_function_helpervar_b2,r9__condX2
        + 1.950111693 sv_function_helpervar_b2,r9__condX3  = 0
 c705:  - aux_cap_helper__b9,r10,band_HIGH_PAIRED__condX1 <= 0
 c706:  - X__b1,r2,band_UNPAIRED + aux_cap_helper__b1,r2,band_UNPAIRED__condX2
        + aux_cap_helper__b1,r2,band_UNPAIRED__condX1  = 0
 c707:  - aux_cap_helper__b4,r11,band_HIGH_PAIRED__condX1 <= 0
 c708:  0.888446608 aux_cap_helper__b7,r9,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b7,r9,band_UNPAIRED__condX1
        - 0.253841888 aux_cap_helper__b7,r9,band_UNPAIRED__Z2
        - aux_cap_b7,r9,band_UNPAIRED  = 0
 c709:  - aux_cap_helper__b4,r10,band_LOW_PAIRED__condX1 <= 0
 c710:  - aux_cap_helper__b3,r4,band_LOW_PAIRED__condX1 <= 0
 c711:  aux_cap_helper__b7,r1,band_UNPAIRED__Z1
        + aux_cap_helper__b7,r1,band_UNPAIRED__Z2  = 1
 c712:  aux_cap_b1,r0,band_UNPAIRED - aux_c_b1,r0
        + aux_cap_b1,r0,band_LOW_PAIRED + aux_cap_b1,r0,band_HIGH_PAIRED  = 0
 c713:  - 3 aux_cap_helper__b8,r2,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b8,r2,band_HIGH_PAIRED__condX3 <= 0
 c714:  aux_cap_helper__b2,r12,band_UNPAIRED__Z2
        + aux_cap_helper__b2,r12,band_UNPAIRED__Z1  = 1
 c715:  aux_cap_helper__b2,r8,band_LOW_PAIRED__Z2
        - aux_cap_helper__b2,r8,band_LOW_PAIRED__condX2 <= 0
 c716:  aux_cap_helper__b1,r6,band_LOW_PAIRED__Z2
        - aux_cap_helper__b1,r6,band_LOW_PAIRED__condX2 <= 0
 c717:  sv_function_helpervar_b7,r13__Z1 + sv_function_helpervar_b7,r13__Z2
        + sv_function_helpervar_b7,r13__Z3  = 1
 c718:  - aux_cap_helper__b7,r5,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b7,r5,band_HIGH_PAIRED__condX1 <= 0
 c719:  aux_cap_helper__b4,r9,band_LOW_PAIRED__condX1
        + aux_cap_helper__b4,r9,band_LOW_PAIRED__condX2
        - X__b4,r9,band_LOW_PAIRED  = 0
 c720:  sv_function_helpervar_b5,r4__condX3
        - 15.8453086887446 sv_function_helpervar_b5,r4__Z3 <= 0
 c721:  - 8.06446672130127 sv_function_helpervar_b9,r2__Z2
        + sv_function_helpervar_b9,r2__condX2 <= 0
 c722:  - aux_cap_helper__b3,r5,band_UNPAIRED__condX1 <= 0
 c723:  GlobalBidder_PSI__i9,k2 - 20668026.8748422 GlobalBidder_hat{W}__i9,k2
        <= 0
 c724:  - aux_cap_helper__b1,r8,band_UNPAIRED__condX1 <= 0
 c725:  - 3 aux_cap_helper__b0,r12,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b0,r12,band_HIGH_PAIRED__condX3 <= 0
 c726:  0.158453086929873 sv_function_helpervar_b0,r7__Z2
        - sv_function_helpervar_b0,r7__condX2 <= 0
 c727:  - 3 aux_cap_helper__b5,r11,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b5,r11,band_HIGH_PAIRED__condX3 <= 0
 c728:  - aux_cap_helper__b8,r8,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b8,r8,band_HIGH_PAIRED__condX1 <= 0
 c729:  - aux_cap_helper__b4,r3,band_LOW_PAIRED__condX1 <= 0
 c730:  aux_cap_helper__b3,r10,band_UNPAIRED__Z2
        + aux_cap_helper__b3,r10,band_UNPAIRED__Z1  = 1
 c731:  0.888446608 aux_cap_helper__b0,r10,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b0,r10,band_UNPAIRED__condX1
        - aux_cap_b0,r10,band_UNPAIRED
        - 0.253841888 aux_cap_helper__b0,r10,band_UNPAIRED__Z2  = 0
 c732:  - 2 aux_cap_helper__b7,r5,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b7,r5,band_HIGH_PAIRED__condX2 <= 0
 c733:  - 0.8112667416 aux_cap_helper__b8,r13,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b8,r13,band_HIGH_PAIRED__Z3
        + 2.0281668543 aux_cap_helper__b8,r13,band_HIGH_PAIRED__condX1
        + 1.8253501688 aux_cap_helper__b8,r13,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b8,r13,band_HIGH_PAIRED__condX2
        - aux_cap_b8,r13,band_HIGH_PAIRED  = 0
 c734:  - sv_function_helpervar_b4,r13__condX1 <= 0
 c735:  aux_cap_helper__b5,r2,band_UNPAIRED__condX1
        - aux_cap_helper__b5,r2,band_UNPAIRED__Z1 <= 0
 c736:  sv_function_helpervar_b5,r1__Z3 + sv_function_helpervar_b5,r1__Z2
        + sv_function_helpervar_b5,r1__Z1  = 1
 c737:  aux_cap_helper__b9,r9,band_UNPAIRED__Z2
        - aux_cap_helper__b9,r9,band_UNPAIRED__condX2 <= 0
 c738:  sv_function_helpervar_b6,r8__condX3
        + sv_function_helpervar_b6,r8__condX1
        + sv_function_helpervar_b6,r8__condX2 - aux_c_b6,r8  = 0
 c739:  - 2 aux_cap_helper__b8,r8,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b8,r8,band_HIGH_PAIRED__condX2 <= 0
 c740:  - aux_cap_helper__b1,r12,band_LOW_PAIRED__condX1 <= 0
 c741:  - X__b8,r5,band_LOW_PAIRED
        + aux_cap_helper__b8,r5,band_LOW_PAIRED__condX2
        + aux_cap_helper__b8,r5,band_LOW_PAIRED__condX1  = 0
 c742:  - aux_cap_helper__b9,r12,band_UNPAIRED__Z1
        + aux_cap_helper__b9,r12,band_UNPAIRED__condX1 <= 0
 c743:  - aux_c_b5,r4 + sv_function_helpervar_b5,r4__condX3
        + sv_function_helpervar_b5,r4__condX2
        + sv_function_helpervar_b5,r4__condX1  = 0
 c744:  aux_cap_helper__b1,r6,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b1,r6,band_HIGH_PAIRED__Z3 <= 0
 c745:  - aux_Omega_b3,r3 + 299.114117595169 aux_sv_b3,r3  = 0
 c746:  - aux_cap_helper__b1,r0,band_UNPAIRED__condX2
        + aux_cap_helper__b1,r0,band_UNPAIRED__Z2 <= 0
 c747:  sv_function_helpervar_b4,r4__condX1
        - 0.158453086903718 sv_function_helpervar_b4,r4__Z1 <= 0
 c748:  X__b5,r6,band_UNPAIRED + X__b4,r6,band_UNPAIRED
        + X__b6,r6,band_UNPAIRED + X__b2,r6,band_UNPAIRED
        + X__b8,r6,band_UNPAIRED + X__b3,r6,band_UNPAIRED
        + X__b7,r6,band_UNPAIRED + X__b1,r6,band_UNPAIRED
        + X__b9,r6,band_UNPAIRED + X__b0,r6,band_UNPAIRED <= 2
 c749:  sv_function_helpervar_b0,r12__condX2
        - 6.1155275836706 sv_function_helpervar_b0,r12__Z2 <= 0
 c750:  0.888446608 aux_cap_helper__b7,r13,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b7,r13,band_UNPAIRED__condX1
        - 0.253841888 aux_cap_helper__b7,r13,band_UNPAIRED__Z2
        - aux_cap_b7,r13,band_UNPAIRED  = 0
 c751:  aux_cap_helper__b2,r10,band_UNPAIRED__Z2
        - aux_cap_helper__b2,r10,band_UNPAIRED__condX2 <= 0
 c752:  sv_function_helpervar_b6,r7__condX2
        - 6.04011244526223 sv_function_helpervar_b6,r7__Z2 <= 0
 c753:  1.8253501688 aux_cap_helper__b3,r11,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b3,r11,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b3,r11,band_HIGH_PAIRED__condX1
        - aux_cap_b3,r11,band_HIGH_PAIRED
        - 0.8112667416 aux_cap_helper__b3,r11,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b3,r11,band_HIGH_PAIRED__Z3  = 0
 c754:  - X__b5,r10,band_HIGH_PAIRED
        + aux_cap_helper__b5,r10,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b5,r10,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b5,r10,band_HIGH_PAIRED__condX2  = 0
 c755:  aux_cap_helper__b1,r2,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b1,r2,band_LOW_PAIRED__Z2 <= 0
 c756:  aux_cap_helper__b0,r3,band_UNPAIRED__Z2
        + aux_cap_helper__b0,r3,band_UNPAIRED__Z1  = 1
 c757:  sv_function_helpervar_b2,r10__condX1
        - 0.158453086903173 sv_function_helpervar_b2,r10__Z1 <= 0
 c758:  aux_cap_helper__b5,r5,band_LOW_PAIRED__Z2
        - aux_cap_helper__b5,r5,band_LOW_PAIRED__condX2 <= 0
 c759:  X__b8,r10,band_HIGH_PAIRED + X__b5,r10,band_HIGH_PAIRED
        + X__b7,r10,band_HIGH_PAIRED + X__b1,r10,band_HIGH_PAIRED
        + X__b9,r10,band_HIGH_PAIRED + X__b4,r10,band_HIGH_PAIRED
        + X__b6,r10,band_HIGH_PAIRED + X__b3,r10,band_HIGH_PAIRED
        + X__b2,r10,band_HIGH_PAIRED + X__b0,r10,band_HIGH_PAIRED <= 3
 c760:  3.1788778092 aux_cap_helper__b5,r0,band_LOW_PAIRED__condX1
        + 4.4504289328 aux_cap_helper__b5,r0,band_LOW_PAIRED__condX2
        - aux_cap_b5,r0,band_LOW_PAIRED
        - 1.2715511236 aux_cap_helper__b5,r0,band_LOW_PAIRED__Z2  = 0
 c761:  - 2 aux_cap_helper__b4,r10,band_UNPAIRED__Z2
        + aux_cap_helper__b4,r10,band_UNPAIRED__condX2 <= 0
 c762:  aux_cap_helper__b3,r4,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b3,r4,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b3,r4,band_HIGH_PAIRED__condX3
        - X__b3,r4,band_HIGH_PAIRED  = 0
 c763:  - aux_cap_helper__b2,r4,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b2,r4,band_HIGH_PAIRED__Z2 <= 0
 c764:  - 15.8453086887446 sv_function_helpervar_b1,r1__Z3
        + sv_function_helpervar_b1,r1__condX3 <= 0
 c765:  aux_cap_helper__b0,r0,band_LOW_PAIRED__Z2
        + aux_cap_helper__b0,r0,band_LOW_PAIRED__Z1  = 1
 c766:  aux_cap_b9,r2,band_LOW_PAIRED + aux_cap_b9,r2,band_UNPAIRED
        + aux_cap_b9,r2,band_HIGH_PAIRED - aux_c_b9,r2  = 0
 c767:  - aux_cap_helper__b1,r11,band_HIGH_PAIRED__condX1 <= 0
 c768:  303.91161400338 sv_function_helpervar_b9,r8__Z2
        + 596.38042585959 sv_function_helpervar_b9,r8__Z3
        + 36.2620578787 sv_function_helpervar_b9,r8__condX3
        + 77.3014046466 sv_function_helpervar_b9,r8__condX2
        + 1995.2925270876 sv_function_helpervar_b9,r8__condX1 - aux_sv_b9,r8
         = 0
 c769:  X__b9,r12,band_LOW_PAIRED - GlobalBidder_W_ir__b9,r12
        + X__b9,r12,band_UNPAIRED + X__b9,r12,band_HIGH_PAIRED >= 0
 c770:  - aux_cap_helper__b6,r3,band_HIGH_PAIRED__condX1 <= 0
 c771:  - sv_function_helpervar_b5,r10__condX3
        + 5.95863360031299 sv_function_helpervar_b5,r10__Z3 <= 0
 c772:  5.91005559387013 sv_function_helpervar_b0,r11__Z3
        - sv_function_helpervar_b0,r11__condX3 <= 0
 c773:  - X__b1,r3,band_HIGH_PAIRED
        + aux_cap_helper__b1,r3,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b1,r3,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b1,r3,band_HIGH_PAIRED__condX1  = 0
 c774:  aux_cap_helper__b9,r9,band_UNPAIRED__Z1
        + aux_cap_helper__b9,r9,band_UNPAIRED__Z2  = 1
 c775:  sv_function_helpervar_b0,r5__Z1 + sv_function_helpervar_b0,r5__Z2
        + sv_function_helpervar_b0,r5__Z3  = 1
 c776:  aux_cap_helper__b1,r2,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b1,r2,band_HIGH_PAIRED__Z2 <= 0
 c777:  - sv_function_helpervar_b7,r8__condX1 <= 0
 c778:  sv_function_helpervar_b3,r7__condX2
        - 5.93657899836289 sv_function_helpervar_b3,r7__Z2 <= 0
 c779:  - aux_cap_helper__b7,r0,band_LOW_PAIRED__Z1
        + aux_cap_helper__b7,r0,band_LOW_PAIRED__condX1 <= 0
 c780:  - 2 aux_cap_helper__b6,r4,band_UNPAIRED__Z2
        + aux_cap_helper__b6,r4,band_UNPAIRED__condX2 <= 0
 c781:  - 15.8453086887446 sv_function_helpervar_b5,r12__Z3
        + sv_function_helpervar_b5,r12__condX3 <= 0
 c782:  - aux_cap_helper__b9,r0,band_HIGH_PAIRED__condX1 <= 0
 c783:  aux_cap_helper__b5,r2,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b5,r2,band_UNPAIRED__Z2 <= 0
 c784:  aux_cap_helper__b3,r2,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b3,r2,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b3,r2,band_HIGH_PAIRED__Z3  = 1
 c785:  - 2 aux_cap_helper__b1,r5,band_UNPAIRED__Z2
        + aux_cap_helper__b1,r5,band_UNPAIRED__condX2 <= 0
 c786:  aux_cap_helper__b3,r2,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b3,r2,band_UNPAIRED__Z2 <= 0
 c787:  - 2 aux_cap_helper__b7,r0,band_LOW_PAIRED__Z2
        + aux_cap_helper__b7,r0,band_LOW_PAIRED__condX2 <= 0
 c788:  - aux_c_b7,r11 + aux_cap_b7,r11,band_HIGH_PAIRED
        + aux_cap_b7,r11,band_LOW_PAIRED + aux_cap_b7,r11,band_UNPAIRED  = 0
 c789:  aux_cap_b1,r3,band_UNPAIRED + aux_cap_b1,r3,band_HIGH_PAIRED
        + aux_cap_b1,r3,band_LOW_PAIRED - aux_c_b1,r3  = 0
 c790:  - aux_cap_helper__b3,r8,band_HIGH_PAIRED__condX1 <= 0
 c791:  - aux_cap_helper__b2,r11,band_LOW_PAIRED__condX1 <= 0
 c792:  - aux_cap_helper__b5,r12,band_LOW_PAIRED__condX1 <= 0
 c793:  - aux_cap_helper__b4,r5,band_UNPAIRED__Z1
        + aux_cap_helper__b4,r5,band_UNPAIRED__condX1 <= 0
 c794:  sv_function_helpervar_b4,r9__condX1
        - 0.158453086968442 sv_function_helpervar_b4,r9__Z1 <= 0
 c795:  sv_function_helpervar_b6,r2__Z3 + sv_function_helpervar_b6,r2__Z1
        + sv_function_helpervar_b6,r2__Z2  = 1
 c796:  - sv_function_helpervar_b1,r2__condX2
        + 0.158453086927277 sv_function_helpervar_b1,r2__Z2 <= 0
 c797:  - 15.8453086887446 sv_function_helpervar_b4,r3__Z3
        + sv_function_helpervar_b4,r3__condX3 <= 0
 c798:  aux_cap_helper__b1,r2,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b1,r2,band_HIGH_PAIRED__Z1 <= 0
 c799:  aux_cap_helper__b0,r4,band_UNPAIRED__condX1
        + aux_cap_helper__b0,r4,band_UNPAIRED__condX2 - X__b0,r4,band_UNPAIRED
         = 0
 c800:  aux_cap_b3,r12,band_LOW_PAIRED + aux_cap_b3,r12,band_HIGH_PAIRED
        - aux_c_b3,r12 + aux_cap_b3,r12,band_UNPAIRED  = 0
 c801:  aux_cap_helper__b8,r11,band_UNPAIRED__Z2
        + aux_cap_helper__b8,r11,band_UNPAIRED__Z1  = 1
 c802:  - 2 aux_cap_helper__b4,r5,band_UNPAIRED__Z2
        + aux_cap_helper__b4,r5,band_UNPAIRED__condX2 <= 0
 c803:  aux_cap_helper__b1,r3,band_UNPAIRED__condX1
        - aux_cap_helper__b1,r3,band_UNPAIRED__Z1 <= 0
 c804:  - aux_cap_helper__b6,r4,band_UNPAIRED__Z1
        + aux_cap_helper__b6,r4,band_UNPAIRED__condX1 <= 0
 c805:  239.3828522569 aux_sv_b8,r8 - aux_Omega_b8,r8  = 0
 c806:  2.4111877663 sv_function_helpervar_b1,r9__condX3
        + 150.9476061572 sv_function_helpervar_b1,r9__condX1
        + 7.0656935931 sv_function_helpervar_b1,r9__condX2 - aux_sv_b1,r9
        + 50.3795935186953 sv_function_helpervar_b1,r9__Z3
        + 22.7985331941569 sv_function_helpervar_b1,r9__Z2  = 0
 c807:  aux_cap_helper__b2,r0,band_UNPAIRED__Z2
        + aux_cap_helper__b2,r0,band_UNPAIRED__Z1  = 1
 c808:  aux_cap_helper__b6,r5,band_UNPAIRED__Z1
        + aux_cap_helper__b6,r5,band_UNPAIRED__Z2  = 1
 c809:  - sv_function_helpervar_b7,r3__condX2
        + 0.158453086899856 sv_function_helpervar_b7,r3__Z2 <= 0
 c810:  aux_cap_helper__b9,r0,band_LOW_PAIRED__condX1
        + aux_cap_helper__b9,r0,band_LOW_PAIRED__condX2
        - X__b9,r0,band_LOW_PAIRED  = 0
 c811:  - aux_cap_helper__b5,r2,band_LOW_PAIRED__condX2
        + aux_cap_helper__b5,r2,band_LOW_PAIRED__Z2 <= 0
 c812:  - aux_cap_helper__b6,r10,band_UNPAIRED__condX1 <= 0
 c813:  X__b9,r7,band_LOW_PAIRED - GlobalBidder_W_ir__b9,r7
        + X__b9,r7,band_HIGH_PAIRED + X__b9,r7,band_UNPAIRED >= 0
 c814:  3870.57593649674 aux_sv_b8,r11 - aux_Omega_b8,r11  = 0
 c815:  - 0.158453086903028 sv_function_helpervar_b9,r8__Z1
        + sv_function_helpervar_b9,r8__condX1 <= 0
 c816:  aux_cap_helper__b1,r3,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b1,r3,band_UNPAIRED__Z2 <= 0
 c817:  aux_cap_helper__b6,r2,band_UNPAIRED__Z2
        - aux_cap_helper__b6,r2,band_UNPAIRED__condX2 <= 0
 c818:  - aux_cap_helper__b7,r4,band_HIGH_PAIRED__condX1 <= 0
 c819:  - X__b8,r9,band_HIGH_PAIRED
        + aux_cap_helper__b8,r9,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b8,r9,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b8,r9,band_HIGH_PAIRED__condX2  = 0
 c820:  - aux_cap_helper__b3,r5,band_HIGH_PAIRED__condX1 <= 0
 c821:  - aux_cap_helper__b1,r5,band_UNPAIRED__Z1
        + aux_cap_helper__b1,r5,band_UNPAIRED__condX1 <= 0
 c822:  aux_cap_helper__b8,r6,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b8,r6,band_LOW_PAIRED__Z2 <= 0
 c823:  3.1788778092 aux_cap_helper__b4,r9,band_LOW_PAIRED__condX1
        + 4.4504289328 aux_cap_helper__b4,r9,band_LOW_PAIRED__condX2
        - aux_cap_b4,r9,band_LOW_PAIRED
        - 1.2715511236 aux_cap_helper__b4,r9,band_LOW_PAIRED__Z2  = 0
 c824:  - aux_cap_helper__b8,r1,band_UNPAIRED__condX2
        + aux_cap_helper__b8,r1,band_UNPAIRED__Z2 <= 0
 c825:  - sv_function_helpervar_b4,r5__condX1 <= 0
 c826:  - aux_cap_helper__b4,r13,band_HIGH_PAIRED__condX1 <= 0
 c827:  - X__b6,r13,band_HIGH_PAIRED
        + aux_cap_helper__b6,r13,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b6,r13,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b6,r13,band_HIGH_PAIRED__condX1  = 0
 c828:  - sv_function_helpervar_b6,r1__condX3
        + 5.56357622273766 sv_function_helpervar_b6,r1__Z3 <= 0
 c829:  aux_cap_helper__b2,r7,band_LOW_PAIRED__Z2
        + aux_cap_helper__b2,r7,band_LOW_PAIRED__Z1  = 1
 c830:  - aux_cap_helper__b6,r7,band_LOW_PAIRED__condX2
        + aux_cap_helper__b6,r7,band_LOW_PAIRED__Z2 <= 0
 c831:  - aux_cap_helper__b1,r12,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b1,r12,band_HIGH_PAIRED__condX1 <= 0
 c832:  6.30568026667668 sv_function_helpervar_b1,r13__Z3
        - sv_function_helpervar_b1,r13__condX3 <= 0
 c833:  sv_function_helpervar_b9,r13__Z2 + sv_function_helpervar_b9,r13__Z3
        + sv_function_helpervar_b9,r13__Z1  = 1
 c834:  aux_cap_helper__b3,r8,band_UNPAIRED__condX2
        + aux_cap_helper__b3,r8,band_UNPAIRED__condX1 - X__b3,r8,band_UNPAIRED
         = 0
 c835:  sv_function_helpervar_b5,r0__condX2
        - 6.0671903614491 sv_function_helpervar_b5,r0__Z2 <= 0
 c836:  - 2 aux_cap_helper__b1,r12,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b1,r12,band_HIGH_PAIRED__condX2 <= 0
 c837:  - aux_cap_helper__b6,r11,band_LOW_PAIRED__condX1 <= 0
 c838:  aux_cap_helper__b6,r10,band_UNPAIRED__condX2
        + aux_cap_helper__b6,r10,band_UNPAIRED__condX1
        - X__b6,r10,band_UNPAIRED  = 0
 c839:  - aux_cap_helper__b3,r10,band_LOW_PAIRED__condX1 <= 0
 c840:  aux_cap_helper__b5,r7,band_LOW_PAIRED__Z2
        + aux_cap_helper__b5,r7,band_LOW_PAIRED__Z1  = 1
 c841:  sv_function_helpervar_b4,r6__Z3 + sv_function_helpervar_b4,r6__Z1
        + sv_function_helpervar_b4,r6__Z2  = 1
 c842:  0.0102040816326531 X__b7,r0,band_UNPAIRED - GlobalBidder_W_ir__b7,r0
        + 0.0102040816326531 X__b7,r0,band_HIGH_PAIRED
        + 0.0102040816326531 X__b7,r0,band_LOW_PAIRED <= 0
 c843:  - X__b9,r1,band_UNPAIRED + aux_cap_helper__b9,r1,band_UNPAIRED__condX1
        + aux_cap_helper__b9,r1,band_UNPAIRED__condX2  = 0
 c844:  - aux_cap_helper__b0,r10,band_LOW_PAIRED__condX1 <= 0
 c845:  - aux_cap_helper__b3,r11,band_LOW_PAIRED__condX1 <= 0
 c846:  - aux_cap_helper__b5,r4,band_UNPAIRED__condX1 <= 0
 c847:  - X__b9,r12,band_LOW_PAIRED
        + aux_cap_helper__b9,r12,band_LOW_PAIRED__condX2
        + aux_cap_helper__b9,r12,band_LOW_PAIRED__condX1  = 0
 c848:  - 3 aux_cap_helper__b3,r12,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b3,r12,band_HIGH_PAIRED__condX3 <= 0
 c849:  - aux_sv_b0,r6 + 54.1402537668113 sv_function_helpervar_b0,r6__Z3
        + 24.7558950360143 sv_function_helpervar_b0,r6__Z2
        + 2.6477396011 sv_function_helpervar_b0,r6__condX3
        + 7.5077195166 sv_function_helpervar_b0,r6__condX2
        + 163.7425742946 sv_function_helpervar_b0,r6__condX1  = 0
 c850:  aux_cap_b4,r6,band_LOW_PAIRED + aux_cap_b4,r6,band_HIGH_PAIRED
        - aux_c_b4,r6 + aux_cap_b4,r6,band_UNPAIRED  = 0
 c851:  - aux_cap_helper__b3,r0,band_UNPAIRED__condX2
        + aux_cap_helper__b3,r0,band_UNPAIRED__Z2 <= 0
 c852:  - 7.52981976069524 sv_function_helpervar_b7,r4__Z2
        + sv_function_helpervar_b7,r4__condX2 <= 0
 c853:  - aux_cap_b5,r11,band_LOW_PAIRED
        - 1.2715511236 aux_cap_helper__b5,r11,band_LOW_PAIRED__Z2
        + 4.4504289328 aux_cap_helper__b5,r11,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b5,r11,band_LOW_PAIRED__condX1  = 0
 c854:  - aux_Omega_b7,r7 + 6065.58661605031 aux_sv_b7,r7  = 0
 c855:  aux_cap_helper__b7,r12,band_UNPAIRED__condX1
        - aux_cap_helper__b7,r12,band_UNPAIRED__Z1 <= 0
 c856:  aux_cap_helper__b8,r8,band_UNPAIRED__condX1
        + aux_cap_helper__b8,r8,band_UNPAIRED__condX2 - X__b8,r8,band_UNPAIRED
         = 0
 c857:  - sv_function_helpervar_b9,r0__condX2
        + 0.158453086901946 sv_function_helpervar_b9,r0__Z2 <= 0
 c858:  aux_cap_helper__b8,r10,band_UNPAIRED__condX2
        + aux_cap_helper__b8,r10,band_UNPAIRED__condX1
        - X__b8,r10,band_UNPAIRED  = 0
 c859:  - aux_cap_helper__b6,r7,band_UNPAIRED__condX1 <= 0
 c860:  14.5397396249 sv_function_helpervar_b6,r9__condX3
        + 46.9721959479 sv_function_helpervar_b6,r9__condX2
        + 942.4308563795 sv_function_helpervar_b6,r9__condX1
        + 322.691405647157 sv_function_helpervar_b6,r9__Z3
        + 141.888188980584 sv_function_helpervar_b6,r9__Z2 - aux_sv_b6,r9  = 0
 c861:  GlobalBidder_W_ir__b9,r11 + GlobalBidder_W_ir__b9,r12
        + GlobalBidder_W_ir__b9,r10 + GlobalBidder_W_ir__b9,r13
        + GlobalBidder_W_ir__b9,r6 + GlobalBidder_W_ir__b9,r7
        + GlobalBidder_W_ir__b9,r8 + GlobalBidder_W_ir__b9,r9
        - GlobalBidder_W_i_9 + GlobalBidder_W_ir__b9,r2
        + GlobalBidder_W_ir__b9,r3 + GlobalBidder_W_ir__b9,r4
        + GlobalBidder_W_ir__b9,r5 + GlobalBidder_W_ir__b9,r0
        + GlobalBidder_W_ir__b9,r1  = 0
 c862:  - 1.2715511236 aux_cap_helper__b6,r6,band_LOW_PAIRED__Z2
        + 4.4504289328 aux_cap_helper__b6,r6,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b6,r6,band_LOW_PAIRED__condX1
        - aux_cap_b6,r6,band_LOW_PAIRED  = 0
 c863:  aux_cap_helper__b7,r12,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b7,r12,band_UNPAIRED__Z2 <= 0
 c864:  aux_cap_helper__b1,r7,band_LOW_PAIRED__Z1
        + aux_cap_helper__b1,r7,band_LOW_PAIRED__Z2  = 1
 c865:  - X__b3,r7,band_LOW_PAIRED
        + aux_cap_helper__b3,r7,band_LOW_PAIRED__condX2
        + aux_cap_helper__b3,r7,band_LOW_PAIRED__condX1  = 0
 c866:  - 1.2715511236 aux_cap_helper__b4,r0,band_LOW_PAIRED__Z2
        - aux_cap_b4,r0,band_LOW_PAIRED
        + 3.1788778092 aux_cap_helper__b4,r0,band_LOW_PAIRED__condX1
        + 4.4504289328 aux_cap_helper__b4,r0,band_LOW_PAIRED__condX2  = 0
 c867:  aux_cap_helper__b4,r1,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b4,r1,band_HIGH_PAIRED__Z3 <= 0
 c868:  sv_function_helpervar_b9,r7__condX3
        - 15.8453086887446 sv_function_helpervar_b9,r7__Z3 <= 0
 c869:  380.466984333677 sv_function_helpervar_b7,r0__Z3
        + 191.114362985836 sv_function_helpervar_b7,r0__Z2 - aux_sv_b7,r0
        + 1255.6694401044 sv_function_helpervar_b7,r0__condX1
        + 49.5435972497 sv_function_helpervar_b7,r0__condX2
        + 22.494942504 sv_function_helpervar_b7,r0__condX3  = 0
 c870:  aux_cap_helper__b8,r4,band_UNPAIRED__condX1
        - aux_cap_helper__b8,r4,band_UNPAIRED__Z1 <= 0
 c871:  - 2 aux_cap_helper__b2,r1,band_LOW_PAIRED__Z2
        + aux_cap_helper__b2,r1,band_LOW_PAIRED__condX2 <= 0
 c872:  - aux_cap_helper__b8,r6,band_UNPAIRED__condX1 <= 0
 c873:  - aux_cap_helper__b4,r7,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b4,r7,band_HIGH_PAIRED__condX1 <= 0
 c874:  aux_cap_helper__b0,r12,band_UNPAIRED__Z2
        - aux_cap_helper__b0,r12,band_UNPAIRED__condX2 <= 0
 c875:  - 15.8453086887446 sv_function_helpervar_b6,r0__Z3
        + sv_function_helpervar_b6,r0__condX3 <= 0
 c876:  0.158453087025632 sv_function_helpervar_b4,r6__Z2
        - sv_function_helpervar_b4,r6__condX2 <= 0
 c877:  - aux_cap_helper__b6,r5,band_HIGH_PAIRED__condX1 <= 0
 c878:  - aux_cap_b1,r2,band_UNPAIRED
        + 0.888446608 aux_cap_helper__b1,r2,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b1,r2,band_UNPAIRED__condX1
        - 0.253841888 aux_cap_helper__b1,r2,band_UNPAIRED__Z2  = 0
 c879:  aux_cap_helper__b8,r7,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b8,r7,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b8,r7,band_HIGH_PAIRED__condX2
        - X__b8,r7,band_HIGH_PAIRED  = 0
 c880:  - aux_cap_helper__b1,r3,band_LOW_PAIRED__condX2
        + aux_cap_helper__b1,r3,band_LOW_PAIRED__Z2 <= 0
 c881:  - aux_c_b7,r5 + aux_cap_b7,r5,band_HIGH_PAIRED
        + aux_cap_b7,r5,band_UNPAIRED + aux_cap_b7,r5,band_LOW_PAIRED  = 0
 c882:  GlobalBidder_W_i_7 - 14 GlobalBidder_hat{W}__i7,k0 >= 0
 c883:  aux_cap_helper__b8,r4,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b8,r4,band_UNPAIRED__Z2 <= 0
 c884:  - aux_cap_helper__b3,r7,band_UNPAIRED__condX1 <= 0
 c885:  - aux_cap_helper__b4,r13,band_LOW_PAIRED__condX1 <= 0
 c886:  - 3 aux_cap_helper__b5,r9,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b5,r9,band_HIGH_PAIRED__condX3 <= 0
 c887:  - aux_cap_helper__b3,r7,band_LOW_PAIRED__condX1 <= 0
 c888:  sv_function_helpervar_b7,r13__condX3
        - 15.8453086887446 sv_function_helpervar_b7,r13__Z3 <= 0
 c889:  - sv_function_helpervar_b9,r4__condX1 <= 0
 c890:  - X__b7,r3,band_LOW_PAIRED
        + aux_cap_helper__b7,r3,band_LOW_PAIRED__condX2
        + aux_cap_helper__b7,r3,band_LOW_PAIRED__condX1  = 0
 c891:  - sv_function_helpervar_b7,r13__condX2
        + 0.158453086902532 sv_function_helpervar_b7,r13__Z2 <= 0
 c892:  aux_cap_helper__b2,r9,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b2,r9,band_HIGH_PAIRED__Z3 <= 0
 c893:  3.1788778092 aux_cap_helper__b3,r13,band_LOW_PAIRED__condX1
        + 4.4504289328 aux_cap_helper__b3,r13,band_LOW_PAIRED__condX2
        - aux_cap_b3,r13,band_LOW_PAIRED
        - 1.2715511236 aux_cap_helper__b3,r13,band_LOW_PAIRED__Z2  = 0
 c894:  - aux_sv_b4,r12 + 133.210376902135 sv_function_helpervar_b4,r12__Z2
        + 298.901345327476 sv_function_helpervar_b4,r12__Z3
        + 42.7099597125 sv_function_helpervar_b4,r12__condX2
        + 13.8549101531 sv_function_helpervar_b4,r12__condX3
        + 883.4028076519 sv_function_helpervar_b4,r12__condX1  = 0
 c895:  aux_cap_helper__b6,r11,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b6,r11,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b6,r11,band_HIGH_PAIRED__condX1
        - X__b6,r11,band_HIGH_PAIRED  = 0
 c896:  0.0102040816326531 X__b8,r7,band_LOW_PAIRED
        + 0.0102040816326531 X__b8,r7,band_UNPAIRED - GlobalBidder_W_ir__b8,r7
        + 0.0102040816326531 X__b8,r7,band_HIGH_PAIRED <= 0
 c897:  - aux_c_b0,r13 + sv_function_helpervar_b0,r13__condX1
        + sv_function_helpervar_b0,r13__condX2
        + sv_function_helpervar_b0,r13__condX3  = 0
 c898:  - 3 aux_cap_helper__b7,r1,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b7,r1,band_HIGH_PAIRED__condX3 <= 0
 c899:  - aux_sv_b0,r0 + 55.1940308087276 sv_function_helpervar_b0,r0__Z3
        + 24.7025981751903 sv_function_helpervar_b0,r0__Z2
        + 2.5812355612 sv_function_helpervar_b0,r0__condX3
        + 7.844076873 sv_function_helpervar_b0,r0__condX2
        + 163.7425743787 sv_function_helpervar_b0,r0__condX1  = 0
 c900:  - aux_cap_b3,r4,band_UNPAIRED
        + 0.63460472 aux_cap_helper__b3,r4,band_UNPAIRED__condX1
        + 0.888446608 aux_cap_helper__b3,r4,band_UNPAIRED__condX2
        - 0.253841888 aux_cap_helper__b3,r4,band_UNPAIRED__Z2  = 0
 c901:  - aux_cap_helper__b5,r11,band_UNPAIRED__condX2
        + aux_cap_helper__b5,r11,band_UNPAIRED__Z2 <= 0
 c902:  sv_function_helpervar_b1,r9__condX3
        - 15.8453086887446 sv_function_helpervar_b1,r9__Z3 <= 0
 c903:  - sv_function_helpervar_b5,r10__condX2
        + 0.158453086987074 sv_function_helpervar_b5,r10__Z2 <= 0
 c904:  X__b4,r13,band_LOW_PAIRED + X__b5,r13,band_LOW_PAIRED
        + X__b8,r13,band_LOW_PAIRED + X__b7,r13,band_LOW_PAIRED
        + X__b6,r13,band_LOW_PAIRED + X__b9,r13,band_LOW_PAIRED
        + X__b2,r13,band_LOW_PAIRED + X__b3,r13,band_LOW_PAIRED
        + X__b1,r13,band_LOW_PAIRED + X__b0,r13,band_LOW_PAIRED <= 2
 c905:  sv_function_helpervar_b8,r2__condX2
        - 6.45410399062127 sv_function_helpervar_b8,r2__Z2 <= 0
 c906:  - aux_cap_helper__b2,r1,band_LOW_PAIRED__Z1
        + aux_cap_helper__b2,r1,band_LOW_PAIRED__condX1 <= 0
 c907:  - 3 aux_cap_helper__b5,r6,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b5,r6,band_HIGH_PAIRED__condX3 <= 0
 c908:  sv_function_helpervar_b2,r9__Z2 + sv_function_helpervar_b2,r9__Z3
        + sv_function_helpervar_b2,r9__Z1  = 1
 c909:  - aux_cap_helper__b0,r8,band_HIGH_PAIRED__condX1 <= 0
 c910:  aux_cap_helper__b7,r7,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b7,r7,band_HIGH_PAIRED__Z1 <= 0
 c911:  - aux_cap_helper__b2,r12,band_LOW_PAIRED__condX1 <= 0
 c912:  aux_cap_helper__b2,r9,band_LOW_PAIRED__Z2
        - aux_cap_helper__b2,r9,band_LOW_PAIRED__condX2 <= 0
 c913:  aux_cap_helper__b3,r5,band_UNPAIRED__condX1
        + aux_cap_helper__b3,r5,band_UNPAIRED__condX2 - X__b3,r5,band_UNPAIRED
         = 0
 c914:  - sv_function_helpervar_b2,r10__condX3
        + 5.98243245052792 sv_function_helpervar_b2,r10__Z3 <= 0
 c915:  X__b7,r9,band_LOW_PAIRED - GlobalBidder_W_ir__b7,r9
        + X__b7,r9,band_UNPAIRED + X__b7,r9,band_HIGH_PAIRED >= 0
 c916:  aux_cap_helper__b6,r0,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b6,r0,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b6,r0,band_HIGH_PAIRED__condX2
        - X__b6,r0,band_HIGH_PAIRED  = 0
 c917:  - aux_cap_b8,r0,band_HIGH_PAIRED
        + 1.8253501688 aux_cap_helper__b8,r0,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b8,r0,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b8,r0,band_HIGH_PAIRED__condX1
        + 1.21690011250031 aux_cap_helper__b8,r0,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b8,r0,band_HIGH_PAIRED__Z2  = 0
 c918:  aux_cap_helper__b9,r2,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b9,r2,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b9,r2,band_HIGH_PAIRED__Z1  = 1
 c919:  aux_cap_helper__b7,r7,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b7,r7,band_HIGH_PAIRED__Z2 <= 0
 c920:  sv_function_helpervar_b9,r6__condX1
        - 0.158453086924069 sv_function_helpervar_b9,r6__Z1 <= 0
 c921:  aux_cap_helper__b2,r5,band_LOW_PAIRED__Z2
        - aux_cap_helper__b2,r5,band_LOW_PAIRED__condX2 <= 0
 c922:  - aux_c_b5,r9 + aux_cap_b5,r9,band_UNPAIRED
        + aux_cap_b5,r9,band_HIGH_PAIRED + aux_cap_b5,r9,band_LOW_PAIRED  = 0
 c923:  - 2 aux_cap_helper__b8,r6,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b8,r6,band_HIGH_PAIRED__condX2 <= 0
 c924:  aux_cap_helper__b9,r5,band_LOW_PAIRED__Z2
        + aux_cap_helper__b9,r5,band_LOW_PAIRED__Z1  = 1
 c925:  X__b0,r4,band_LOW_PAIRED + X__b7,r4,band_LOW_PAIRED
        + X__b4,r4,band_LOW_PAIRED + X__b5,r4,band_LOW_PAIRED
        + X__b3,r4,band_LOW_PAIRED + X__b1,r4,band_LOW_PAIRED
        + X__b2,r4,band_LOW_PAIRED + X__b8,r4,band_LOW_PAIRED
        + X__b6,r4,band_LOW_PAIRED + X__b9,r4,band_LOW_PAIRED <= 2
 c926:  aux_cap_helper__b3,r2,band_UNPAIRED__condX1
        - aux_cap_helper__b3,r2,band_UNPAIRED__Z1 <= 0
 c927:  6.27092287796977 sv_function_helpervar_b9,r12__Z3
        - sv_function_helpervar_b9,r12__condX3 <= 0
 c928:  - X__b3,r6,band_LOW_PAIRED
        + aux_cap_helper__b3,r6,band_LOW_PAIRED__condX1
        + aux_cap_helper__b3,r6,band_LOW_PAIRED__condX2  = 0
 c929:  - 2 aux_cap_helper__b4,r7,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b4,r7,band_HIGH_PAIRED__condX2 <= 0
 c930:  - aux_cap_helper__b4,r0,band_LOW_PAIRED__condX1 <= 0
 c931:  sv_function_helpervar_b4,r4__condX2
        - 6.21753278941992 sv_function_helpervar_b4,r4__Z2 <= 0
 c932:  - 6.19715991137678 sv_function_helpervar_b2,r6__Z2
        + sv_function_helpervar_b2,r6__condX2 <= 0
 c933:  - aux_cap_helper__b9,r3,band_UNPAIRED__condX1 <= 0
 c934:  sv_function_helpervar_b1,r7__condX2
        + sv_function_helpervar_b1,r7__condX1
        + sv_function_helpervar_b1,r7__condX3 - aux_c_b1,r7  = 0
 c935:  - 7.25480608867014 sv_function_helpervar_b8,r3__Z2
        + sv_function_helpervar_b8,r3__condX2 <= 0
 c936:  - 3 aux_cap_helper__b7,r3,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b7,r3,band_HIGH_PAIRED__condX3 <= 0
 c937:  sv_function_helpervar_b6,r13__condX1
        - 0.15845308692958 sv_function_helpervar_b6,r13__Z1 <= 0
 c938:  - aux_cap_helper__b2,r7,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b2,r7,band_HIGH_PAIRED__Z3 <= 0
 c939:  aux_cap_helper__b4,r0,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b4,r0,band_HIGH_PAIRED__Z1 <= 0
 c940:  - aux_cap_helper__b6,r11,band_UNPAIRED__condX1 <= 0
 c941:  aux_cap_helper__b4,r0,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b4,r0,band_HIGH_PAIRED__Z2 <= 0
 c942:  sv_function_helpervar_b7,r2__condX1
        - 0.158453086902389 sv_function_helpervar_b7,r2__Z1 <= 0
 c943:  aux_cap_helper__b7,r4,band_LOW_PAIRED__condX1
        - aux_cap_helper__b7,r4,band_LOW_PAIRED__Z1 <= 0
 c944:  - aux_cap_b1,r1,band_LOW_PAIRED
        - 1.2715511236 aux_cap_helper__b1,r1,band_LOW_PAIRED__Z2
        + 4.4504289328 aux_cap_helper__b1,r1,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b1,r1,band_LOW_PAIRED__condX1  = 0
 c945:  - 15.8453086887446 sv_function_helpervar_b7,r1__Z3
        + sv_function_helpervar_b7,r1__condX3 <= 0
 c946:  sv_function_helpervar_b3,r6__condX1
        + sv_function_helpervar_b3,r6__condX2
        + sv_function_helpervar_b3,r6__condX3 - aux_c_b3,r6  = 0
 c947:  - aux_cap_helper__b5,r13,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b5,r13,band_HIGH_PAIRED__condX1 <= 0
 c948:  aux_cap_helper__b1,r5,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b1,r5,band_LOW_PAIRED__Z2 <= 0
 c949:  - X__b0,r9,band_UNPAIRED + aux_cap_helper__b0,r9,band_UNPAIRED__condX1
        + aux_cap_helper__b0,r9,band_UNPAIRED__condX2  = 0
 c950:  - aux_cap_helper__b2,r7,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b2,r7,band_HIGH_PAIRED__Z2 <= 0
 c951:  - 2 aux_cap_helper__b5,r13,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b5,r13,band_HIGH_PAIRED__condX2 <= 0
 c952:  sv_function_helpervar_b6,r3__condX1
        - 0.158453086945764 sv_function_helpervar_b6,r3__Z1 <= 0
 c953:  - aux_sv_b5,r5 + 16.0246557815 sv_function_helpervar_b5,r5__condX3
        + 43.9893697025 sv_function_helpervar_b5,r5__condX2
        + 978.90187304 sv_function_helpervar_b5,r5__condX1
        + 148.139772139765 sv_function_helpervar_b5,r5__Z2
        + 320.565951252757 sv_function_helpervar_b5,r5__Z3  = 0
 c954:  - aux_cap_helper__b1,r13,band_LOW_PAIRED__condX1 <= 0
 c955:  - aux_cap_helper__b8,r1,band_LOW_PAIRED__condX2
        + aux_cap_helper__b8,r1,band_LOW_PAIRED__Z2 <= 0
 c956:  5.57220532880487 sv_function_helpervar_b0,r9__Z3
        - sv_function_helpervar_b0,r9__condX3 <= 0
 c957:  aux_cap_helper__b4,r11,band_LOW_PAIRED__Z2
        + aux_cap_helper__b4,r11,band_LOW_PAIRED__Z1  = 1
 c958:  - aux_cap_helper__b8,r7,band_UNPAIRED__condX1 <= 0
 c959:  - X__b9,r8,band_LOW_PAIRED
        + aux_cap_helper__b9,r8,band_LOW_PAIRED__condX2
        + aux_cap_helper__b9,r8,band_LOW_PAIRED__condX1  = 0
 c960:  - aux_cap_b7,r6,band_LOW_PAIRED
        - 1.2715511236 aux_cap_helper__b7,r6,band_LOW_PAIRED__Z2
        + 3.1788778092 aux_cap_helper__b7,r6,band_LOW_PAIRED__condX1
        + 4.4504289328 aux_cap_helper__b7,r6,band_LOW_PAIRED__condX2  = 0
 c961:  - X__b8,r13,band_LOW_PAIRED
        + aux_cap_helper__b8,r13,band_LOW_PAIRED__condX2
        + aux_cap_helper__b8,r13,band_LOW_PAIRED__condX1  = 0
 c962:  sv_function_helpervar_b1,r8__Z3 + sv_function_helpervar_b1,r8__Z2
        + sv_function_helpervar_b1,r8__Z1  = 1
 c963:  - sv_function_helpervar_b7,r7__condX1 <= 0
 c964:  - aux_c_b7,r4 + sv_function_helpervar_b7,r4__condX1
        + sv_function_helpervar_b7,r4__condX2
        + sv_function_helpervar_b7,r4__condX3  = 0
 c965:  aux_cap_helper__b3,r3,band_UNPAIRED__condX1
        - aux_cap_helper__b3,r3,band_UNPAIRED__Z1 <= 0
 c966:  - 2 aux_cap_helper__b6,r5,band_UNPAIRED__Z2
        + aux_cap_helper__b6,r5,band_UNPAIRED__condX2 <= 0
 c967:  - aux_cap_b6,r1,band_HIGH_PAIRED
        - 0.8112667416 aux_cap_helper__b6,r1,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b6,r1,band_HIGH_PAIRED__Z3
        + 1.8253501688 aux_cap_helper__b6,r1,band_HIGH_PAIRED__condX3
        + 2.0281668543 aux_cap_helper__b6,r1,band_HIGH_PAIRED__condX1
        + 2.8394335959 aux_cap_helper__b6,r1,band_HIGH_PAIRED__condX2  = 0
 c968:  - 6.28882502139831 sv_function_helpervar_b3,r4__Z2
        + sv_function_helpervar_b3,r4__condX2 <= 0
 c969:  - sv_function_helpervar_b9,r13__condX3
        + 6.67287044758352 sv_function_helpervar_b9,r13__Z3 <= 0
 c970:  - aux_Omega_b7,r8 + 312.596738075745 aux_sv_b7,r8  = 0
 c971:  - aux_cap_helper__b9,r9,band_LOW_PAIRED__condX1 <= 0
 c972:  - aux_cap_helper__b3,r1,band_LOW_PAIRED__condX1 <= 0
 c973:  0.0102040816326531 X__b8,r12,band_UNPAIRED
        + 0.0102040816326531 X__b8,r12,band_HIGH_PAIRED
        - GlobalBidder_W_ir__b8,r12
        + 0.0102040816326531 X__b8,r12,band_LOW_PAIRED <= 0
 c974:  aux_cap_helper__b1,r5,band_LOW_PAIRED__condX1
        - aux_cap_helper__b1,r5,band_LOW_PAIRED__Z1 <= 0
 c975:  aux_cap_helper__b1,r9,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b1,r9,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b1,r9,band_HIGH_PAIRED__condX1
        - X__b1,r9,band_HIGH_PAIRED  = 0
 c976:  - aux_Omega_b3,r6 + 32.160116419942 aux_sv_b3,r6  = 0
 c977:  - 6.04618933446524 sv_function_helpervar_b0,r6__Z2
        + sv_function_helpervar_b0,r6__condX2 <= 0
 c978:  aux_cap_helper__b3,r3,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b3,r3,band_UNPAIRED__Z2 <= 0
 c979:  aux_cap_helper__b2,r9,band_UNPAIRED__Z1
        + aux_cap_helper__b2,r9,band_UNPAIRED__Z2  = 1
 c980:  - sv_function_helpervar_b6,r2__condX1 <= 0
 c981:  sv_function_helpervar_b2,r13__Z2 + sv_function_helpervar_b2,r13__Z1
        + sv_function_helpervar_b2,r13__Z3  = 1
 c982:  - sv_function_helpervar_b3,r8__condX1 <= 0
 c983:  - aux_cap_helper__b4,r9,band_LOW_PAIRED__condX1 <= 0
 c984:  aux_cap_helper__b2,r2,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b2,r2,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b2,r2,band_HIGH_PAIRED__Z2  = 1
 c985:  - aux_cap_helper__b3,r10,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b3,r10,band_HIGH_PAIRED__Z2 <= 0
 c986:  X__b9,r8,band_HIGH_PAIRED + X__b2,r8,band_HIGH_PAIRED
        + X__b1,r8,band_HIGH_PAIRED + X__b7,r8,band_HIGH_PAIRED
        + X__b0,r8,band_HIGH_PAIRED + X__b8,r8,band_HIGH_PAIRED
        + X__b6,r8,band_HIGH_PAIRED + X__b3,r8,band_HIGH_PAIRED
        + X__b4,r8,band_HIGH_PAIRED + X__b5,r8,band_HIGH_PAIRED <= 3
 c987:  - aux_cap_helper__b4,r6,band_HIGH_PAIRED__condX1 <= 0
 c988:  sv_function_helpervar_b2,r2__condX1
        - 0.158453086894398 sv_function_helpervar_b2,r2__Z1 <= 0
 c989:  - 15.8453086887446 sv_function_helpervar_b1,r10__Z3
        + sv_function_helpervar_b1,r10__condX3 <= 0
 c990:  - aux_cap_helper__b0,r5,band_UNPAIRED__condX1 <= 0
 c991:  - aux_cap_helper__b6,r0,band_LOW_PAIRED__condX1 <= 0
 c992:  aux_cap_helper__b7,r1,band_UNPAIRED__condX1
        - aux_cap_helper__b7,r1,band_UNPAIRED__Z1 <= 0
 c993:  - 6.05183208365148 sv_function_helpervar_b0,r7__Z2
        + sv_function_helpervar_b0,r7__condX2 <= 0
 c994:  - aux_cap_b1,r12,band_HIGH_PAIRED
        + 1.21690011250031 aux_cap_helper__b1,r12,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b1,r12,band_HIGH_PAIRED__Z2
        + 1.8253501688 aux_cap_helper__b1,r12,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b1,r12,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b1,r12,band_HIGH_PAIRED__condX1  = 0
 c995:  - 1.2715511236 aux_cap_helper__b7,r8,band_LOW_PAIRED__Z2
        + 4.4504289328 aux_cap_helper__b7,r8,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b7,r8,band_LOW_PAIRED__condX1
        - aux_cap_b7,r8,band_LOW_PAIRED  = 0
 c996:  aux_cap_helper__b7,r1,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b7,r1,band_UNPAIRED__Z2 <= 0
 c997:  - 3 aux_cap_helper__b5,r4,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b5,r4,band_HIGH_PAIRED__condX3 <= 0
 c998:  6.60787655101763 sv_function_helpervar_b2,r13__Z3
        - sv_function_helpervar_b2,r13__condX3 <= 0
 c999:  - aux_cap_helper__b9,r3,band_HIGH_PAIRED__condX1 <= 0
 c1000: aux_cap_helper__b2,r3,band_UNPAIRED__Z2
        - aux_cap_helper__b2,r3,band_UNPAIRED__condX2 <= 0
 c1001: aux_cap_helper__b1,r3,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b1,r3,band_HIGH_PAIRED__Z3 <= 0
 c1002: - aux_c_b7,r11 + sv_function_helpervar_b7,r11__condX3
        + sv_function_helpervar_b7,r11__condX2
        + sv_function_helpervar_b7,r11__condX1  = 0
 c1003: aux_cap_helper__b9,r10,band_UNPAIRED__condX1
        - aux_cap_helper__b9,r10,band_UNPAIRED__Z1 <= 0
 c1004: - aux_cap_helper__b9,r13,band_HIGH_PAIRED__condX1 <= 0
 c1005: 0.158453086938541 sv_function_helpervar_b4,r11__Z2
        - sv_function_helpervar_b4,r11__condX2 <= 0
 c1006: - aux_Omega_b3,r0 + 60.2853396762374 aux_sv_b3,r0  = 0
 c1007: aux_cap_helper__b6,r13,band_LOW_PAIRED__Z2
        + aux_cap_helper__b6,r13,band_LOW_PAIRED__Z1  = 1
 c1008: 0.888446608 aux_cap_helper__b4,r4,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b4,r4,band_UNPAIRED__condX1
        - aux_cap_b4,r4,band_UNPAIRED
        - 0.253841888 aux_cap_helper__b4,r4,band_UNPAIRED__Z2  = 0
 c1009: - sv_function_helpervar_b0,r11__condX1 <= 0
 c1010: aux_cap_helper__b9,r10,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b9,r10,band_UNPAIRED__Z2 <= 0
 c1011: aux_cap_helper__b9,r4,band_LOW_PAIRED__condX2
        + aux_cap_helper__b9,r4,band_LOW_PAIRED__condX1
        - X__b9,r4,band_LOW_PAIRED  = 0
 c1012: aux_cap_helper__b8,r3,band_UNPAIRED__condX1
        - aux_cap_helper__b8,r3,band_UNPAIRED__Z1 <= 0
 c1013: sv_function_helpervar_b1,r12__condX1
        - 0.158453086906349 sv_function_helpervar_b1,r12__Z1 <= 0
 c1014: - aux_Omega_b1,r1 + 359.794440244774 aux_sv_b1,r1  = 0
 c1015: sv_function_helpervar_b1,r6__Z1 + sv_function_helpervar_b1,r6__Z3
        + sv_function_helpervar_b1,r6__Z2  = 1
 c1016: aux_cap_helper__b3,r11,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b3,r11,band_HIGH_PAIRED__Z1 <= 0
 c1017: aux_cap_b4,r10,band_HIGH_PAIRED + aux_cap_b4,r10,band_LOW_PAIRED
        - aux_c_b4,r10 + aux_cap_b4,r10,band_UNPAIRED  = 0
 c1018: aux_cap_helper__b5,r13,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b5,r13,band_LOW_PAIRED__Z2 <= 0
 c1019: - sv_function_helpervar_b5,r4__condX2
        + 0.158453086914415 sv_function_helpervar_b5,r4__Z2 <= 0
 c1020: - 15.8453086887446 sv_function_helpervar_b9,r8__Z3
        + sv_function_helpervar_b9,r8__condX3 <= 0
 c1021: - aux_cap_helper__b2,r2,band_LOW_PAIRED__condX2
        + aux_cap_helper__b2,r2,band_LOW_PAIRED__Z2 <= 0
 c1022: 1.8253501688 aux_cap_helper__b8,r3,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b8,r3,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b8,r3,band_HIGH_PAIRED__condX1
        + 1.21690011250031 aux_cap_helper__b8,r3,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b8,r3,band_HIGH_PAIRED__Z2
        - aux_cap_b8,r3,band_HIGH_PAIRED  = 0
 c1023: aux_cap_b1,r12,band_HIGH_PAIRED + aux_cap_b1,r12,band_UNPAIRED
        + aux_cap_b1,r12,band_LOW_PAIRED - aux_c_b1,r12  = 0
 c1024: 0.63460472 aux_cap_helper__b0,r2,band_UNPAIRED__condX1
        + 0.888446608 aux_cap_helper__b0,r2,band_UNPAIRED__condX2
        - 0.253841888 aux_cap_helper__b0,r2,band_UNPAIRED__Z2
        - aux_cap_b0,r2,band_UNPAIRED  = 0
 c1025: aux_cap_helper__b1,r0,band_LOW_PAIRED__Z2
        - aux_cap_helper__b1,r0,band_LOW_PAIRED__condX2 <= 0
 c1026: aux_cap_helper__b1,r5,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b1,r5,band_HIGH_PAIRED__Z2 <= 0
 c1027: aux_cap_helper__b3,r11,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b3,r11,band_HIGH_PAIRED__Z2 <= 0
 c1028: - sv_function_helpervar_b3,r4__condX1 <= 0
 c1029: - aux_cap_helper__b6,r5,band_UNPAIRED__Z1
        + aux_cap_helper__b6,r5,band_UNPAIRED__condX1 <= 0
 c1030: - 2 aux_cap_helper__b2,r4,band_LOW_PAIRED__Z2
        + aux_cap_helper__b2,r4,band_LOW_PAIRED__condX2 <= 0
 c1031: 5.83047977299225 sv_function_helpervar_b1,r10__Z3
        - sv_function_helpervar_b1,r10__condX3 <= 0
 c1032: - sv_function_helpervar_b4,r5__condX2
        + 0.158453086918754 sv_function_helpervar_b4,r5__Z2 <= 0
 c1033: aux_cap_helper__b6,r5,band_LOW_PAIRED__Z1
        + aux_cap_helper__b6,r5,band_LOW_PAIRED__Z2  = 1
 c1034: 0.63460472 aux_cap_helper__b9,r0,band_UNPAIRED__condX1
        + 0.888446608 aux_cap_helper__b9,r0,band_UNPAIRED__condX2
        - aux_cap_b9,r0,band_UNPAIRED
        - 0.253841888 aux_cap_helper__b9,r0,band_UNPAIRED__Z2  = 0
 c1035: - aux_cap_helper__b7,r10,band_UNPAIRED__condX2
        + aux_cap_helper__b7,r10,band_UNPAIRED__Z2 <= 0
 c1036: - sv_function_helpervar_b7,r6__condX3
        + 7.48311679006299 sv_function_helpervar_b7,r6__Z3 <= 0
 c1037: - aux_cap_helper__b8,r13,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b8,r13,band_HIGH_PAIRED__condX1 <= 0
 c1038: 0.158453086963932 sv_function_helpervar_b1,r8__Z2
        - sv_function_helpervar_b1,r8__condX2 <= 0
 c1039: aux_cap_helper__b3,r6,band_UNPAIRED__Z2
        + aux_cap_helper__b3,r6,band_UNPAIRED__Z1  = 1
 c1040: aux_cap_helper__b2,r4,band_UNPAIRED__condX2
        + aux_cap_helper__b2,r4,band_UNPAIRED__condX1 - X__b2,r4,band_UNPAIRED
         = 0
 c1041: - X__b7,r13,band_UNPAIRED
        + aux_cap_helper__b7,r13,band_UNPAIRED__condX2
        + aux_cap_helper__b7,r13,band_UNPAIRED__condX1  = 0
 c1042: sv_function_helpervar_b8,r6__condX2
        - 6.10570058921676 sv_function_helpervar_b8,r6__Z2 <= 0
 c1043: 0.158453086937505 sv_function_helpervar_b0,r9__Z2
        - sv_function_helpervar_b0,r9__condX2 <= 0
 c1044: - 2 aux_cap_helper__b8,r13,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b8,r13,band_HIGH_PAIRED__condX2 <= 0
 c1045: sv_function_helpervar_b2,r8__condX2
        - 5.78536513589384 sv_function_helpervar_b2,r8__Z2 <= 0
 c1046: - aux_cap_helper__b6,r10,band_HIGH_PAIRED__condX1 <= 0
 c1047: sv_function_helpervar_b3,r7__condX1
        - 0.158453086911699 sv_function_helpervar_b3,r7__Z1 <= 0
 c1048: aux_cap_helper__b8,r6,band_LOW_PAIRED__condX1
        - aux_cap_helper__b8,r6,band_LOW_PAIRED__Z1 <= 0
 c1049: 0.158453086928019 sv_function_helpervar_b0,r13__Z2
        - sv_function_helpervar_b0,r13__condX2 <= 0
 c1050: aux_cap_helper__b2,r4,band_LOW_PAIRED__condX1
        + aux_cap_helper__b2,r4,band_LOW_PAIRED__condX2
        - X__b2,r4,band_LOW_PAIRED  = 0
 c1051: aux_cap_helper__b1,r1,band_UNPAIRED__Z2
        - aux_cap_helper__b1,r1,band_UNPAIRED__condX2 <= 0
 c1052: - aux_cap_helper__b4,r5,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b4,r5,band_HIGH_PAIRED__condX1 <= 0
 c1053: aux_cap_helper__b5,r13,band_LOW_PAIRED__condX1
        - aux_cap_helper__b5,r13,band_LOW_PAIRED__Z1 <= 0
 c1054: aux_cap_helper__b4,r11,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b4,r11,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b4,r11,band_HIGH_PAIRED__Z1  = 1
 c1055: - aux_cap_b4,r1,band_HIGH_PAIRED
        + 2.0281668543 aux_cap_helper__b4,r1,band_HIGH_PAIRED__condX1
        + 2.8394335959 aux_cap_helper__b4,r1,band_HIGH_PAIRED__condX2
        + 1.8253501688 aux_cap_helper__b4,r1,band_HIGH_PAIRED__condX3
        + 1.21690011250031 aux_cap_helper__b4,r1,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b4,r1,band_HIGH_PAIRED__Z2  = 0
 c1056: aux_cap_helper__b1,r5,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b1,r5,band_HIGH_PAIRED__Z1 <= 0
 c1057: aux_cap_helper__b1,r9,band_LOW_PAIRED__Z2
        + aux_cap_helper__b1,r9,band_LOW_PAIRED__Z1  = 1
 c1058: aux_cap_helper__b7,r4,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b7,r4,band_LOW_PAIRED__Z2 <= 0
 c1059: - aux_cap_helper__b1,r7,band_UNPAIRED__condX1 <= 0
 c1060: - aux_cap_helper__b3,r3,band_HIGH_PAIRED__condX1 <= 0
 c1061: - sv_function_helpervar_b2,r7__condX1 <= 0
 c1062: - aux_cap_b1,r3,band_LOW_PAIRED
        + 4.4504289328 aux_cap_helper__b1,r3,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b1,r3,band_LOW_PAIRED__condX1
        - 1.2715511236 aux_cap_helper__b1,r3,band_LOW_PAIRED__Z2  = 0
 c1063: - aux_cap_helper__b2,r4,band_LOW_PAIRED__Z1
        + aux_cap_helper__b2,r4,band_LOW_PAIRED__condX1 <= 0
 c1064: - aux_cap_helper__b7,r2,band_LOW_PAIRED__condX1 <= 0
 c1065: - aux_cap_helper__b6,r12,band_LOW_PAIRED__Z1
        + aux_cap_helper__b6,r12,band_LOW_PAIRED__condX1 <= 0
 c1066: aux_cap_helper__b7,r13,band_UNPAIRED__Z2
        + aux_cap_helper__b7,r13,band_UNPAIRED__Z1  = 1
 c1067: - sv_function_helpervar_b5,r9__condX2
        + 0.158453086965282 sv_function_helpervar_b5,r9__Z2 <= 0
 c1068: - aux_cap_b9,r10,band_UNPAIRED
        + 0.888446608 aux_cap_helper__b9,r10,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b9,r10,band_UNPAIRED__condX1
        - 0.253841888 aux_cap_helper__b9,r10,band_UNPAIRED__Z2  = 0
 c1069: aux_cap_helper__b5,r13,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b5,r13,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b5,r13,band_HIGH_PAIRED__condX1
        - X__b5,r13,band_HIGH_PAIRED  = 0
 c1070: - aux_cap_helper__b7,r7,band_LOW_PAIRED__Z1
        + aux_cap_helper__b7,r7,band_LOW_PAIRED__condX1 <= 0
 c1071: - 2 aux_cap_helper__b6,r12,band_LOW_PAIRED__Z2
        + aux_cap_helper__b6,r12,band_LOW_PAIRED__condX2 <= 0
 c1072: - 6.63929351348566 sv_function_helpervar_b2,r12__Z2
        + sv_function_helpervar_b2,r12__condX2 <= 0
 c1073: sv_function_helpervar_b8,r10__condX2
        - 8.23774537961124 sv_function_helpervar_b8,r10__Z2 <= 0
 c1074: - X__b4,r2,band_LOW_PAIRED
        + aux_cap_helper__b4,r2,band_LOW_PAIRED__condX2
        + aux_cap_helper__b4,r2,band_LOW_PAIRED__condX1  = 0
 c1075: - aux_cap_helper__b2,r2,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b2,r2,band_HIGH_PAIRED__Z2 <= 0
 c1076: - 2 aux_cap_helper__b7,r7,band_LOW_PAIRED__Z2
        + aux_cap_helper__b7,r7,band_LOW_PAIRED__condX2 <= 0
 c1077: - aux_cap_helper__b8,r13,band_LOW_PAIRED__Z1
        + aux_cap_helper__b8,r13,band_LOW_PAIRED__condX1 <= 0
 c1078: sv_function_helpervar_b5,r7__Z3 + sv_function_helpervar_b5,r7__Z1
        + sv_function_helpervar_b5,r7__Z2  = 1
 c1079: sv_function_helpervar_b8,r10__Z1 + sv_function_helpervar_b8,r10__Z2
        + sv_function_helpervar_b8,r10__Z3  = 1
 c1080: aux_cap_helper__b6,r12,band_LOW_PAIRED__condX2
        + aux_cap_helper__b6,r12,band_LOW_PAIRED__condX1
        - X__b6,r12,band_LOW_PAIRED  = 0
 c1081: aux_cap_helper__b2,r7,band_LOW_PAIRED__condX1
        - aux_cap_helper__b2,r7,band_LOW_PAIRED__Z1 <= 0
 c1082: 0.158453086960565 sv_function_helpervar_b5,r3__Z2
        - sv_function_helpervar_b5,r3__condX2 <= 0
 c1083: - 2 aux_cap_helper__b8,r13,band_LOW_PAIRED__Z2
        + aux_cap_helper__b8,r13,band_LOW_PAIRED__condX2 <= 0
 c1084: - aux_cap_helper__b4,r11,band_LOW_PAIRED__Z1
        + aux_cap_helper__b4,r11,band_LOW_PAIRED__condX1 <= 0
 c1085: aux_cap_helper__b5,r3,band_UNPAIRED__Z2
        + aux_cap_helper__b5,r3,band_UNPAIRED__Z1  = 1
 c1086: aux_cap_helper__b9,r8,band_UNPAIRED__Z1
        + aux_cap_helper__b9,r8,band_UNPAIRED__Z2  = 1
 c1087: aux_cap_helper__b2,r12,band_UNPAIRED__condX1
        - aux_cap_helper__b2,r12,band_UNPAIRED__Z1 <= 0
 c1088: aux_cap_helper__b4,r7,band_UNPAIRED__condX1
        + aux_cap_helper__b4,r7,band_UNPAIRED__condX2 - X__b4,r7,band_UNPAIRED
         = 0
 c1089: aux_cap_helper__b5,r3,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b5,r3,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b5,r3,band_HIGH_PAIRED__condX2
        - X__b5,r3,band_HIGH_PAIRED  = 0
 c1090: sv_function_helpervar_b8,r7__condX3
        - 15.8453086887446 sv_function_helpervar_b8,r7__Z3 <= 0
 c1091: aux_cap_helper__b7,r1,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b7,r1,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b7,r1,band_HIGH_PAIRED__Z1  = 1
 c1092: 1.8253501688 aux_cap_helper__b2,r9,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b2,r9,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b2,r9,band_HIGH_PAIRED__condX1
        - 0.8112667416 aux_cap_helper__b2,r9,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b2,r9,band_HIGH_PAIRED__Z3
        - aux_cap_b2,r9,band_HIGH_PAIRED  = 0
 c1093: - 2 aux_cap_helper__b5,r8,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b5,r8,band_HIGH_PAIRED__condX2 <= 0
 c1094: - 2 aux_cap_helper__b4,r5,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b4,r5,band_HIGH_PAIRED__condX2 <= 0
 c1095: - aux_cap_helper__b4,r13,band_UNPAIRED__condX1 <= 0
 c1096: 4.4504289328 aux_cap_helper__b7,r4,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b7,r4,band_LOW_PAIRED__condX1
        - aux_cap_b7,r4,band_LOW_PAIRED
        - 1.2715511236 aux_cap_helper__b7,r4,band_LOW_PAIRED__Z2  = 0
 c1097: aux_cap_helper__b2,r8,band_LOW_PAIRED__Z2
        + aux_cap_helper__b2,r8,band_LOW_PAIRED__Z1  = 1
 c1098: aux_cap_helper__b1,r2,band_LOW_PAIRED__condX1
        - aux_cap_helper__b1,r2,band_LOW_PAIRED__Z1 <= 0
 c1099: 39.3592561157 sv_function_helpervar_b4,r4__condX2
        + 883.4028077066 sv_function_helpervar_b4,r4__condX1
        + 14.5389655227 sv_function_helpervar_b4,r4__condX3
        + 133.741306230755 sv_function_helpervar_b4,r4__Z2
        + 288.06227683566 sv_function_helpervar_b4,r4__Z3 - aux_sv_b4,r4  = 0
 c1100: aux_cap_helper__b8,r2,band_UNPAIRED__Z1
        + aux_cap_helper__b8,r2,band_UNPAIRED__Z2  = 1
 c1101: - aux_cap_helper__b1,r6,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b1,r6,band_HIGH_PAIRED__Z3 <= 0
 c1102: aux_cap_helper__b8,r3,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b8,r3,band_UNPAIRED__Z2 <= 0
 c1103: - sv_function_helpervar_b1,r4__condX2
        + 0.158453086904274 sv_function_helpervar_b1,r4__Z2 <= 0
 c1104: aux_cap_helper__b9,r12,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b9,r12,band_LOW_PAIRED__Z2 <= 0
 c1105: sv_function_helpervar_b3,r7__condX3
        - 15.8453086887446 sv_function_helpervar_b3,r7__Z3 <= 0
 c1106: sv_function_helpervar_b3,r10__Z3 + sv_function_helpervar_b3,r10__Z1
        + sv_function_helpervar_b3,r10__Z2  = 1
 c1107: - 3 aux_cap_helper__b8,r4,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b8,r4,band_HIGH_PAIRED__condX3 <= 0
 c1108: aux_cap_helper__b4,r1,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b4,r1,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b4,r1,band_HIGH_PAIRED__Z1  = 1
 c1109: aux_cap_helper__b3,r12,band_LOW_PAIRED__condX1
        - aux_cap_helper__b3,r12,band_LOW_PAIRED__Z1 <= 0
 c1110: X__b7,r3,band_LOW_PAIRED - GlobalBidder_W_ir__b7,r3
        + X__b7,r3,band_UNPAIRED + X__b7,r3,band_HIGH_PAIRED >= 0
 c1111: aux_cap_helper__b5,r2,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b5,r2,band_HIGH_PAIRED__condX2 <= 0
 c1112: - 7.02238717159649 sv_function_helpervar_b8,r1__Z2
        + sv_function_helpervar_b8,r1__condX2 <= 0
 c1113: - 6.29672555527569 sv_function_helpervar_b4,r1__Z2
        + sv_function_helpervar_b4,r1__condX2 <= 0
 c1114: - aux_cap_helper__b1,r10,band_UNPAIRED__condX1 <= 0
 c1115: aux_cap_helper__b7,r11,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b7,r11,band_HIGH_PAIRED__condX2 <= 0
 c1116: - aux_cap_b1,r10,band_UNPAIRED
        - 0.253841888 aux_cap_helper__b1,r10,band_UNPAIRED__Z2
        + 0.888446608 aux_cap_helper__b1,r10,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b1,r10,band_UNPAIRED__condX1  = 0
 c1117: - 0.158453086896389 sv_function_helpervar_b7,r0__Z1
        + sv_function_helpervar_b7,r0__condX1 <= 0
 c1118: aux_cap_helper__b8,r7,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b8,r7,band_HIGH_PAIRED__condX2 <= 0
 c1119: aux_cap_helper__b3,r12,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b3,r12,band_LOW_PAIRED__Z2 <= 0
 c1120: X__b8,r1,band_LOW_PAIRED + X__b8,r1,band_HIGH_PAIRED
        + X__b8,r1,band_UNPAIRED - GlobalBidder_W_ir__b8,r1 >= 0
 c1121: X__b8,r4,band_HIGH_PAIRED + X__b8,r4,band_UNPAIRED
        + X__b8,r4,band_LOW_PAIRED - GlobalBidder_W_ir__b8,r4 >= 0
 c1122: - aux_cap_b2,r4,band_HIGH_PAIRED
        + 2.0281668543 aux_cap_helper__b2,r4,band_HIGH_PAIRED__condX1
        + 2.8394335959 aux_cap_helper__b2,r4,band_HIGH_PAIRED__condX2
        + 1.8253501688 aux_cap_helper__b2,r4,band_HIGH_PAIRED__condX3
        - 0.8112667416 aux_cap_helper__b2,r4,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b2,r4,band_HIGH_PAIRED__Z3  = 0
 c1123: - aux_cap_helper__b8,r12,band_HIGH_PAIRED__condX1 <= 0
 c1124: 2 aux_cap_helper__b5,r2,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b5,r2,band_HIGH_PAIRED__condX3 <= 0
 c1125: - aux_cap_helper__b7,r11,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b7,r11,band_HIGH_PAIRED__Z3 <= 0
 c1126: - aux_cap_helper__b5,r8,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b5,r8,band_HIGH_PAIRED__condX1 <= 0
 c1127: aux_cap_helper__b4,r8,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b4,r8,band_HIGH_PAIRED__Z3 <= 0
 c1128: - aux_cap_helper__b2,r2,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b2,r2,band_HIGH_PAIRED__Z3 <= 0
 c1129: X__b9,r2,band_UNPAIRED + X__b9,r2,band_HIGH_PAIRED
        - GlobalBidder_W_ir__b9,r2 + X__b9,r2,band_LOW_PAIRED >= 0
 c1130: 2 aux_cap_helper__b8,r7,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b8,r7,band_HIGH_PAIRED__condX3 <= 0
 c1131: 0.158453086929204 sv_function_helpervar_b5,r11__Z2
        - sv_function_helpervar_b5,r11__condX2 <= 0
 c1132: - aux_cap_helper__b9,r6,band_LOW_PAIRED__condX1 <= 0
 c1133: - aux_cap_helper__b1,r6,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b1,r6,band_HIGH_PAIRED__Z2 <= 0
 c1134: aux_cap_helper__b9,r12,band_LOW_PAIRED__condX1
        - aux_cap_helper__b9,r12,band_LOW_PAIRED__Z1 <= 0
 c1135: - 3 aux_cap_helper__b1,r8,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b1,r8,band_HIGH_PAIRED__condX3 <= 0
 c1136: - sv_function_helpervar_b9,r10__condX1 <= 0
 c1137: aux_cap_helper__b0,r1,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b0,r1,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b0,r1,band_HIGH_PAIRED__Z2  = 1
 c1138: aux_cap_helper__b3,r13,band_LOW_PAIRED__condX1
        + aux_cap_helper__b3,r13,band_LOW_PAIRED__condX2
        - X__b3,r13,band_LOW_PAIRED  = 0
 c1139: sv_function_helpervar_b6,r13__condX1
        + sv_function_helpervar_b6,r13__condX2
        + sv_function_helpervar_b6,r13__condX3 - aux_c_b6,r13  = 0
 c1140: aux_cap_helper__b4,r7,band_UNPAIRED__Z2
        + aux_cap_helper__b4,r7,band_UNPAIRED__Z1  = 1
 c1141: aux_cap_helper__b5,r10,band_LOW_PAIRED__condX1
        - aux_cap_helper__b5,r10,band_LOW_PAIRED__Z1 <= 0
 c1142: - aux_cap_helper__b3,r10,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b3,r10,band_HIGH_PAIRED__Z3 <= 0
 c1143: aux_cap_helper__b5,r10,band_LOW_PAIRED__condX1
        + aux_cap_helper__b5,r10,band_LOW_PAIRED__condX2
        - X__b5,r10,band_LOW_PAIRED  = 0
 c1144: aux_cap_helper__b5,r6,band_LOW_PAIRED__Z2
        + aux_cap_helper__b5,r6,band_LOW_PAIRED__Z1  = 1
 c1145: aux_cap_helper__b8,r2,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b8,r2,band_HIGH_PAIRED__condX2 <= 0
 c1146: - 6.1718003564743 sv_function_helpervar_b6,r0__Z2
        + sv_function_helpervar_b6,r0__condX2 <= 0
 c1147: aux_cap_helper__b5,r0,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b5,r0,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b5,r0,band_HIGH_PAIRED__Z3  = 1
 c1148: X__b6,r0,band_LOW_PAIRED + X__b8,r0,band_LOW_PAIRED
        + X__b1,r0,band_LOW_PAIRED + X__b2,r0,band_LOW_PAIRED
        + X__b3,r0,band_LOW_PAIRED + X__b5,r0,band_LOW_PAIRED
        + X__b4,r0,band_LOW_PAIRED + X__b7,r0,band_LOW_PAIRED
        + X__b9,r0,band_LOW_PAIRED + X__b0,r0,band_LOW_PAIRED <= 2
 c1149: - aux_cap_helper__b4,r6,band_LOW_PAIRED__condX1 <= 0
 c1150: 7.4203323054 sv_function_helpervar_b0,r12__condX2
        + 2.6666084355 sv_function_helpervar_b0,r12__condX3
        + 163.7425743912 sv_function_helpervar_b0,r12__condX1
        + 24.7697418100838 sv_function_helpervar_b0,r12__Z2
        + 53.8412712615991 sv_function_helpervar_b0,r12__Z3 - aux_sv_b0,r12
         = 0
 c1151: sv_function_helpervar_b9,r6__Z2 + sv_function_helpervar_b9,r6__Z3
        + sv_function_helpervar_b9,r6__Z1  = 1
 c1152: - aux_cap_b0,r7,band_HIGH_PAIRED
        + 1.8253501688 aux_cap_helper__b0,r7,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b0,r7,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b0,r7,band_HIGH_PAIRED__condX1
        + 1.21690011250031 aux_cap_helper__b0,r7,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b0,r7,band_HIGH_PAIRED__Z2  = 0
 c1153: aux_cap_b9,r7,band_LOW_PAIRED + aux_cap_b9,r7,band_UNPAIRED
        + aux_cap_b9,r7,band_HIGH_PAIRED - aux_c_b9,r7  = 0
 c1154: aux_cap_helper__b2,r13,band_UNPAIRED__Z1
        + aux_cap_helper__b2,r13,band_UNPAIRED__Z2  = 1
 c1155: aux_cap_helper__b5,r10,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b5,r10,band_LOW_PAIRED__Z2 <= 0
 c1156: sv_function_helpervar_b9,r13__condX3
        - 15.8453086887446 sv_function_helpervar_b9,r13__Z3 <= 0
 c1157: aux_cap_helper__b1,r8,band_UNPAIRED__condX1
        + aux_cap_helper__b1,r8,band_UNPAIRED__condX2 - X__b1,r8,band_UNPAIRED
         = 0
 c1158: - aux_c_b8,r3 + sv_function_helpervar_b8,r3__condX3
        + sv_function_helpervar_b8,r3__condX2
        + sv_function_helpervar_b8,r3__condX1  = 0
 c1159: - aux_sv_b6,r3 + 43.4472706273 sv_function_helpervar_b6,r3__condX2
        + 15.1896367915 sv_function_helpervar_b6,r3__condX3
        + 942.4308563987 sv_function_helpervar_b6,r3__condX1
        + 312.393584426085 sv_function_helpervar_b6,r3__Z3
        + 142.446724279051 sv_function_helpervar_b6,r3__Z2  = 0
 c1160: aux_cap_helper__b7,r11,band_LOW_PAIRED__condX1
        - aux_cap_helper__b7,r11,band_LOW_PAIRED__Z1 <= 0
 c1161: sv_function_helpervar_b5,r5__condX2
        - 6.16584813276107 sv_function_helpervar_b5,r5__Z2 <= 0
 c1162: - sv_function_helpervar_b6,r9__condX2
        + 0.158453086948978 sv_function_helpervar_b6,r9__Z2 <= 0
 c1163: - aux_cap_helper__b0,r8,band_LOW_PAIRED__condX1 <= 0
 c1164: - sv_function_helpervar_b0,r1__condX1 <= 0
 c1165: - aux_cap_helper__b1,r13,band_HIGH_PAIRED__condX1 <= 0
 c1166: - 5.56014354927208 sv_function_helpervar_b2,r5__Z2
        + sv_function_helpervar_b2,r5__condX2 <= 0
 c1167: - aux_c_b6,r13 + aux_cap_b6,r13,band_LOW_PAIRED
        + aux_cap_b6,r13,band_UNPAIRED + aux_cap_b6,r13,band_HIGH_PAIRED  = 0
 c1168: aux_cap_helper__b6,r0,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b6,r0,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b6,r0,band_HIGH_PAIRED__Z1  = 1
 c1169: - X__b1,r6,band_HIGH_PAIRED
        + aux_cap_helper__b1,r6,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b1,r6,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b1,r6,band_HIGH_PAIRED__condX1  = 0
 c1170: - aux_cap_b7,r6,band_UNPAIRED
        + 0.888446608 aux_cap_helper__b7,r6,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b7,r6,band_UNPAIRED__condX1
        - 0.253841888 aux_cap_helper__b7,r6,band_UNPAIRED__Z2  = 0
 c1171: aux_cap_helper__b1,r11,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b1,r11,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b1,r11,band_HIGH_PAIRED__condX3
        - X__b1,r11,band_HIGH_PAIRED  = 0
 c1172: - 2 aux_cap_helper__b1,r0,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b1,r0,band_HIGH_PAIRED__condX2 <= 0
 c1173: - X__b5,r0,band_UNPAIRED + aux_cap_helper__b5,r0,band_UNPAIRED__condX2
        + aux_cap_helper__b5,r0,band_UNPAIRED__condX1  = 0
 c1174: aux_cap_helper__b5,r12,band_UNPAIRED__Z2
        - aux_cap_helper__b5,r12,band_UNPAIRED__condX2 <= 0
 c1175: aux_cap_helper__b9,r4,band_LOW_PAIRED__Z2
        + aux_cap_helper__b9,r4,band_LOW_PAIRED__Z1  = 1
 c1176: aux_cap_b0,r8,band_HIGH_PAIRED + aux_cap_b0,r8,band_UNPAIRED
        - aux_c_b0,r8 + aux_cap_b0,r8,band_LOW_PAIRED  = 0
 c1177: - 15.8453086887446 sv_function_helpervar_b3,r11__Z3
        + sv_function_helpervar_b3,r11__condX3 <= 0
 c1178: - aux_cap_helper__b6,r8,band_UNPAIRED__condX1 <= 0
 c1179: aux_cap_helper__b7,r13,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b7,r13,band_UNPAIRED__Z2 <= 0
 c1180: aux_cap_helper__b6,r1,band_UNPAIRED__Z2
        - aux_cap_helper__b6,r1,band_UNPAIRED__condX2 <= 0
 c1181: aux_cap_helper__b2,r12,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b2,r12,band_UNPAIRED__Z2 <= 0
 c1182: - aux_cap_helper__b7,r0,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b7,r0,band_HIGH_PAIRED__condX1 <= 0
 c1183: - sv_function_helpervar_b8,r8__condX1 <= 0
 c1184: - X__b7,r6,band_UNPAIRED + aux_cap_helper__b7,r6,band_UNPAIRED__condX2
        + aux_cap_helper__b7,r6,band_UNPAIRED__condX1  = 0
 c1185: aux_cap_helper__b5,r12,band_UNPAIRED__condX2
        + aux_cap_helper__b5,r12,band_UNPAIRED__condX1
        - X__b5,r12,band_UNPAIRED  = 0
 c1186: aux_cap_helper__b8,r1,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b8,r1,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b8,r1,band_HIGH_PAIRED__condX3
        - X__b8,r1,band_HIGH_PAIRED  = 0
 c1187: aux_cap_helper__b4,r9,band_UNPAIRED__Z2
        - aux_cap_helper__b4,r9,band_UNPAIRED__condX2 <= 0
 c1188: - aux_cap_b0,r2,band_HIGH_PAIRED
        + 2.8394335959 aux_cap_helper__b0,r2,band_HIGH_PAIRED__condX2
        + 1.8253501688 aux_cap_helper__b0,r2,band_HIGH_PAIRED__condX3
        + 2.0281668543 aux_cap_helper__b0,r2,band_HIGH_PAIRED__condX1
        - 0.8112667416 aux_cap_helper__b0,r2,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b0,r2,band_HIGH_PAIRED__Z3  = 0
 c1189: - aux_cap_helper__b5,r5,band_UNPAIRED__condX1 <= 0
 c1190: aux_cap_helper__b9,r0,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b9,r0,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b9,r0,band_HIGH_PAIRED__Z1  = 1
 c1191: - aux_cap_helper__b1,r0,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b1,r0,band_HIGH_PAIRED__condX1 <= 0
 c1192: - sv_function_helpervar_b4,r13__condX2
        + 0.158453086947509 sv_function_helpervar_b4,r13__Z2 <= 0
 c1193: aux_cap_helper__b7,r11,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b7,r11,band_LOW_PAIRED__Z2 <= 0
 c1194: - sv_function_helpervar_b2,r12__condX1 <= 0
 c1195: aux_cap_helper__b3,r11,band_UNPAIRED__Z1
        + aux_cap_helper__b3,r11,band_UNPAIRED__Z2  = 1
 c1196: sv_function_helpervar_b3,r12__Z1 + sv_function_helpervar_b3,r12__Z2
        + sv_function_helpervar_b3,r12__Z3  = 1
 c1197: 4.4504289328 aux_cap_helper__b7,r11,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b7,r11,band_LOW_PAIRED__condX1
        - aux_cap_b7,r11,band_LOW_PAIRED
        - 1.2715511236 aux_cap_helper__b7,r11,band_LOW_PAIRED__Z2  = 0
 c1198: aux_cap_helper__b7,r13,band_UNPAIRED__condX1
        - aux_cap_helper__b7,r13,band_UNPAIRED__Z1 <= 0
 c1199: aux_cap_b5,r1,band_HIGH_PAIRED - aux_c_b5,r1
        + aux_cap_b5,r1,band_UNPAIRED + aux_cap_b5,r1,band_LOW_PAIRED  = 0
 c1200: - 0.158453086891007 sv_function_helpervar_b8,r5__Z1
        + sv_function_helpervar_b8,r5__condX1 <= 0
 c1201: 0.158453086950738 sv_function_helpervar_b4,r10__Z2
        - sv_function_helpervar_b4,r10__condX2 <= 0
 c1202: - sv_function_helpervar_b5,r1__condX1 <= 0
 c1203: aux_cap_helper__b2,r7,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b2,r7,band_LOW_PAIRED__Z2 <= 0
 c1204: aux_cap_helper__b0,r11,band_UNPAIRED__Z2
        - aux_cap_helper__b0,r11,band_UNPAIRED__condX2 <= 0
 c1205: aux_cap_b9,r11,band_HIGH_PAIRED + aux_cap_b9,r11,band_LOW_PAIRED
        + aux_cap_b9,r11,band_UNPAIRED - aux_c_b9,r11  = 0
 c1206: - X__b0,r10,band_LOW_PAIRED
        + aux_cap_helper__b0,r10,band_LOW_PAIRED__condX2
        + aux_cap_helper__b0,r10,band_LOW_PAIRED__condX1  = 0
 c1207: - 2 aux_cap_helper__b4,r11,band_LOW_PAIRED__Z2
        + aux_cap_helper__b4,r11,band_LOW_PAIRED__condX2 <= 0
 c1208: aux_cap_helper__b5,r8,band_LOW_PAIRED__Z2
        - aux_cap_helper__b5,r8,band_LOW_PAIRED__condX2 <= 0
 c1209: aux_cap_helper__b6,r8,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b6,r8,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b6,r8,band_HIGH_PAIRED__condX1
        - X__b6,r8,band_HIGH_PAIRED  = 0
 c1210: - 3 aux_cap_helper__b1,r10,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b1,r10,band_HIGH_PAIRED__condX3 <= 0
 c1211: - 15.8453086887446 sv_function_helpervar_b2,r6__Z3
        + sv_function_helpervar_b2,r6__condX3 <= 0
 c1212: aux_cap_helper__b0,r4,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b0,r4,band_HIGH_PAIRED__Z2 <= 0
 c1213: sv_function_helpervar_b2,r7__Z1 + sv_function_helpervar_b2,r7__Z2
        + sv_function_helpervar_b2,r7__Z3  = 1
 c1214: - aux_c_b2,r13 + sv_function_helpervar_b2,r13__condX1
        + sv_function_helpervar_b2,r13__condX2
        + sv_function_helpervar_b2,r13__condX3  = 0
 c1215: - aux_cap_helper__b3,r1,band_UNPAIRED__Z1
        + aux_cap_helper__b3,r1,band_UNPAIRED__condX1 <= 0
 c1216: aux_cap_helper__b4,r6,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b4,r6,band_UNPAIRED__Z2 <= 0
 c1217: sv_function_helpervar_b5,r0__condX3
        - 15.8453086887446 sv_function_helpervar_b5,r0__Z3 <= 0
 c1218: - aux_cap_helper__b7,r8,band_UNPAIRED__condX2
        + aux_cap_helper__b7,r8,band_UNPAIRED__Z2 <= 0
 c1219: 74.2774459981 sv_function_helpervar_b8,r6__condX2
        + 1636.3575441393 sv_function_helpervar_b8,r6__condX1
        + 26.621800537 sv_function_helpervar_b8,r6__condX3 - aux_sv_b8,r6
        + 538.48751609989 sv_function_helpervar_b8,r6__Z3
        + 247.516413528563 sv_function_helpervar_b8,r6__Z2  = 0
 c1220: aux_cap_helper__b8,r10,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b8,r10,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b8,r10,band_HIGH_PAIRED__Z1  = 1
 c1221: aux_cap_helper__b7,r8,band_LOW_PAIRED__condX2
        + aux_cap_helper__b7,r8,band_LOW_PAIRED__condX1
        - X__b7,r8,band_LOW_PAIRED  = 0
 c1222: sv_function_helpervar_b8,r4__Z3 + sv_function_helpervar_b8,r4__Z2
        + sv_function_helpervar_b8,r4__Z1  = 1
 c1223: 0.17678 aux_Omega_b3,r11 + 0.07433 aux_Omega_b3,r12
        + 0.17678 aux_Omega_b3,r13 + 0.42045 aux_Omega_b3,r10 - v__3
        + 0.42045 aux_Omega_b3,r8 + aux_Omega_b3,r9 + 0.17678 aux_Omega_b3,r6
        + 0.17678 aux_Omega_b3,r7 + 0.07433 aux_Omega_b3,r4
        + 0.17678 aux_Omega_b3,r5 + 0.03125 aux_Omega_b3,r2
        + 0.07433 aux_Omega_b3,r3 + 0.07433 aux_Omega_b3,r0
        + 0.03125 aux_Omega_b3,r1  = 0
 c1224: - 2 aux_cap_helper__b3,r2,band_LOW_PAIRED__Z2
        + aux_cap_helper__b3,r2,band_LOW_PAIRED__condX2 <= 0
 c1225: sv_function_helpervar_b6,r13__Z3 + sv_function_helpervar_b6,r13__Z2
        + sv_function_helpervar_b6,r13__Z1  = 1
 c1226: - aux_cap_helper__b3,r9,band_LOW_PAIRED__condX1 <= 0
 c1227: aux_cap_helper__b6,r3,band_UNPAIRED__Z2
        - aux_cap_helper__b6,r3,band_UNPAIRED__condX2 <= 0
 c1228: aux_cap_helper__b2,r9,band_UNPAIRED__Z2
        - aux_cap_helper__b2,r9,band_UNPAIRED__condX2 <= 0
 c1229: aux_cap_helper__b9,r13,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b9,r13,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b9,r13,band_HIGH_PAIRED__condX3
        - X__b9,r13,band_HIGH_PAIRED  = 0
 c1230: aux_cap_helper__b8,r12,band_UNPAIRED__Z2
        - aux_cap_helper__b8,r12,band_UNPAIRED__condX2 <= 0
 c1231: - X__b8,r5,band_UNPAIRED + aux_cap_helper__b8,r5,band_UNPAIRED__condX2
        + aux_cap_helper__b8,r5,band_UNPAIRED__condX1  = 0
 c1232: - 1.2715511236 aux_cap_helper__b5,r8,band_LOW_PAIRED__Z2
        - aux_cap_b5,r8,band_LOW_PAIRED
        + 3.1788778092 aux_cap_helper__b5,r8,band_LOW_PAIRED__condX1
        + 4.4504289328 aux_cap_helper__b5,r8,band_LOW_PAIRED__condX2  = 0
 c1233: aux_cap_helper__b1,r11,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b1,r11,band_HIGH_PAIRED__Z3 <= 0
 c1234: - X__b8,r12,band_HIGH_PAIRED
        + aux_cap_helper__b8,r12,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b8,r12,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b8,r12,band_HIGH_PAIRED__condX2  = 0
 c1235: aux_cap_helper__b2,r10,band_LOW_PAIRED__Z1
        + aux_cap_helper__b2,r10,band_LOW_PAIRED__Z2  = 1
 c1236: 752.504782623478 aux_sv_b8,r3 - aux_Omega_b8,r3  = 0
 c1237: aux_cap_helper__b4,r1,band_UNPAIRED__condX2
        + aux_cap_helper__b4,r1,band_UNPAIRED__condX1 - X__b4,r1,band_UNPAIRED
         = 0
 c1238: - 2 aux_cap_helper__b5,r10,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b5,r10,band_HIGH_PAIRED__condX2 <= 0
 c1239: aux_cap_helper__b9,r5,band_UNPAIRED__condX1
        - aux_cap_helper__b9,r5,band_UNPAIRED__Z1 <= 0
 c1240: - sv_function_helpervar_b3,r11__condX1 <= 0
 c1241: aux_cap_helper__b8,r11,band_LOW_PAIRED__condX1
        - X__b8,r11,band_LOW_PAIRED
        + aux_cap_helper__b8,r11,band_LOW_PAIRED__condX2  = 0
 c1242: aux_cap_helper__b1,r4,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b1,r4,band_LOW_PAIRED__Z2 <= 0
 c1243: - aux_cap_helper__b3,r0,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b3,r0,band_HIGH_PAIRED__condX1 <= 0
 c1244: - X__b1,r0,band_UNPAIRED + aux_cap_helper__b1,r0,band_UNPAIRED__condX2
        + aux_cap_helper__b1,r0,band_UNPAIRED__condX1  = 0
 c1245: - X__b1,r0,band_HIGH_PAIRED
        + aux_cap_helper__b1,r0,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b1,r0,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b1,r0,band_HIGH_PAIRED__condX1  = 0
 c1246: 0.03125 aux_Omega_b4,r0 + 0.42045 aux_Omega_b4,r9
        + 0.17678 aux_Omega_b4,r8 + 0.07433 aux_Omega_b4,r7
        + 0.07433 aux_Omega_b4,r6 + 0.07433 aux_Omega_b4,r5
        + 0.03125 aux_Omega_b4,r4 + 0.03125 aux_Omega_b4,r3
        + 0.01314 aux_Omega_b4,r2 + 0.01314 aux_Omega_b4,r1 - v__4
        + 0.17678 aux_Omega_b4,r12 + 0.42045 aux_Omega_b4,r13
        + aux_Omega_b4,r10 + 0.42045 aux_Omega_b4,r11  = 0
 c1247: - aux_cap_helper__b5,r6,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b5,r6,band_HIGH_PAIRED__condX1 <= 0
 c1248: aux_cap_helper__b5,r0,band_LOW_PAIRED__condX1
        - aux_cap_helper__b5,r0,band_LOW_PAIRED__Z1 <= 0
 c1249: - sv_function_helpervar_b6,r12__condX3
        + 5.69088093223458 sv_function_helpervar_b6,r12__Z3 <= 0
 c1250: sv_function_helpervar_b7,r8__Z2 + sv_function_helpervar_b7,r8__Z3
        + sv_function_helpervar_b7,r8__Z1  = 1
 c1251: aux_cap_helper__b9,r8,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b9,r8,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b9,r8,band_HIGH_PAIRED__Z1  = 1
 c1252: - aux_cap_helper__b8,r0,band_UNPAIRED__Z1
        + aux_cap_helper__b8,r0,band_UNPAIRED__condX1 <= 0
 c1253: - aux_Omega_b0,r11 + 1521.69526623285 aux_sv_b0,r11  = 0
 c1254: - aux_Omega_b7,r11 + 4200.86248283357 aux_sv_b7,r11  = 0
 c1255: - X__b6,r1,band_HIGH_PAIRED
        + aux_cap_helper__b6,r1,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b6,r1,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b6,r1,band_HIGH_PAIRED__condX2  = 0
 c1256: aux_cap_helper__b3,r4,band_LOW_PAIRED__condX2
        + aux_cap_helper__b3,r4,band_LOW_PAIRED__condX1
        - X__b3,r4,band_LOW_PAIRED  = 0
 c1257: aux_cap_helper__b2,r12,band_LOW_PAIRED__condX2
        + aux_cap_helper__b2,r12,band_LOW_PAIRED__condX1
        - X__b2,r12,band_LOW_PAIRED  = 0
 c1258: aux_cap_helper__b0,r2,band_LOW_PAIRED__Z1
        + aux_cap_helper__b0,r2,band_LOW_PAIRED__Z2  = 1
 c1259: aux_cap_helper__b4,r8,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b4,r8,band_LOW_PAIRED__Z2 <= 0
 c1260: aux_cap_helper__b4,r12,band_UNPAIRED__Z1
        + aux_cap_helper__b4,r12,band_UNPAIRED__Z2  = 1
 c1261: - aux_c_b5,r12 + aux_cap_b5,r12,band_HIGH_PAIRED
        + aux_cap_b5,r12,band_UNPAIRED + aux_cap_b5,r12,band_LOW_PAIRED  = 0
 c1262: 2 aux_cap_helper__b8,r11,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b8,r11,band_HIGH_PAIRED__condX3 <= 0
 c1263: aux_cap_helper__b2,r11,band_UNPAIRED__condX1
        - aux_cap_helper__b2,r11,band_UNPAIRED__Z1 <= 0
 c1264: 0.0102040816326531 X__b7,r7,band_UNPAIRED
        + 0.0102040816326531 X__b7,r7,band_LOW_PAIRED
        - GlobalBidder_W_ir__b7,r7
        + 0.0102040816326531 X__b7,r7,band_HIGH_PAIRED <= 0
 c1265: - aux_cap_b9,r12,band_UNPAIRED
        - 0.253841888 aux_cap_helper__b9,r12,band_UNPAIRED__Z2
        + 0.63460472 aux_cap_helper__b9,r12,band_UNPAIRED__condX1
        + 0.888446608 aux_cap_helper__b9,r12,band_UNPAIRED__condX2  = 0
 c1266: - 15.8453086887446 sv_function_helpervar_b0,r10__Z3
        + sv_function_helpervar_b0,r10__condX3 <= 0
 c1267: 0.0102040816326531 X__b9,r2,band_UNPAIRED
        + 0.0102040816326531 X__b9,r2,band_HIGH_PAIRED
        - GlobalBidder_W_ir__b9,r2
        + 0.0102040816326531 X__b9,r2,band_LOW_PAIRED <= 0
 c1268: - aux_cap_helper__b6,r10,band_LOW_PAIRED__condX1 <= 0
 c1269: - aux_cap_helper__b7,r8,band_UNPAIRED__condX1 <= 0
 c1270: 1.8108654166 sv_function_helpervar_b2,r2__condX3
        + 5.4821408469 sv_function_helpervar_b2,r2__condX2
        + 114.7166939454 sv_function_helpervar_b2,r2__condX1 - aux_sv_b2,r2
        + 38.6292943101178 sv_function_helpervar_b2,r2__Z3
        + 17.3085521339874 sv_function_helpervar_b2,r2__Z2  = 0
 c1271: - aux_cap_helper__b4,r3,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b4,r3,band_HIGH_PAIRED__Z3 <= 0
 c1272: - aux_Omega_b6,r9 + 178.904700123338 aux_sv_b6,r9  = 0
 c1273: 46.3890233127617 aux_sv_b8,r6 - aux_Omega_b8,r6  = 0
 c1274: 2 aux_cap_helper__b6,r9,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b6,r9,band_HIGH_PAIRED__condX3 <= 0
 c1275: aux_cap_b6,r8,band_LOW_PAIRED + aux_cap_b6,r8,band_UNPAIRED
        - aux_c_b6,r8 + aux_cap_b6,r8,band_HIGH_PAIRED  = 0
 c1276: - 15.8453086887446 sv_function_helpervar_b3,r12__Z3
        + sv_function_helpervar_b3,r12__condX3 <= 0
 c1277: - 0.158453086983578 sv_function_helpervar_b2,r5__Z1
        + sv_function_helpervar_b2,r5__condX1 <= 0
 c1278: aux_cap_helper__b5,r4,band_LOW_PAIRED__Z2
        + aux_cap_helper__b5,r4,band_LOW_PAIRED__Z1  = 1
 c1279: - aux_cap_helper__b6,r3,band_UNPAIRED__condX1 <= 0
 c1280: aux_cap_helper__b9,r6,band_LOW_PAIRED__Z2
        - aux_cap_helper__b9,r6,band_LOW_PAIRED__condX2 <= 0
 c1281: aux_cap_helper__b9,r11,band_LOW_PAIRED__condX1
        - aux_cap_helper__b9,r11,band_LOW_PAIRED__Z1 <= 0
 c1282: - aux_cap_helper__b8,r12,band_UNPAIRED__condX1 <= 0
 c1283: - 2 aux_cap_helper__b6,r6,band_LOW_PAIRED__Z2
        + aux_cap_helper__b6,r6,band_LOW_PAIRED__condX2 <= 0
 c1284: - X__b5,r6,band_HIGH_PAIRED
        + aux_cap_helper__b5,r6,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b5,r6,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b5,r6,band_HIGH_PAIRED__condX1  = 0
 c1285: aux_cap_helper__b7,r1,band_LOW_PAIRED__condX1
        + aux_cap_helper__b7,r1,band_LOW_PAIRED__condX2
        - X__b7,r1,band_LOW_PAIRED  = 0
 c1286: 2 aux_cap_helper__b7,r0,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b7,r0,band_HIGH_PAIRED__condX3 <= 0
 c1287: aux_cap_helper__b5,r8,band_UNPAIRED__Z2
        + aux_cap_helper__b5,r8,band_UNPAIRED__Z1  = 1
 c1288: - sv_function_helpervar_b0,r9__condX1 <= 0
 c1289: aux_cap_b0,r11,band_UNPAIRED - aux_c_b0,r11
        + aux_cap_b0,r11,band_HIGH_PAIRED + aux_cap_b0,r11,band_LOW_PAIRED  = 0
 c1290: - aux_cap_helper__b9,r6,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b9,r6,band_HIGH_PAIRED__Z3 <= 0
 c1291: aux_cap_helper__b9,r8,band_UNPAIRED__condX2
        + aux_cap_helper__b9,r8,band_UNPAIRED__condX1 - X__b9,r8,band_UNPAIRED
         = 0
 c1292: aux_cap_helper__b0,r5,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b0,r5,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b0,r5,band_HIGH_PAIRED__condX2
        - X__b0,r5,band_HIGH_PAIRED  = 0
 c1293: - 5.43414635547902 sv_function_helpervar_b6,r4__Z2
        + sv_function_helpervar_b6,r4__condX2 <= 0
 c1294: aux_cap_helper__b0,r7,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b0,r7,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b0,r7,band_HIGH_PAIRED__Z1  = 1
 c1295: sv_function_helpervar_b9,r13__condX3
        + sv_function_helpervar_b9,r13__condX2
        + sv_function_helpervar_b9,r13__condX1 - aux_c_b9,r13  = 0
 c1296: sv_function_helpervar_b5,r1__condX2
        - 5.53380541572838 sv_function_helpervar_b5,r1__Z2 <= 0
 c1297: - 2 aux_cap_helper__b0,r12,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b0,r12,band_HIGH_PAIRED__condX2 <= 0
 c1298: - 2 aux_cap_helper__b5,r11,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b5,r11,band_HIGH_PAIRED__condX2 <= 0
 c1299: - aux_cap_b9,r9,band_LOW_PAIRED
        + 3.1788778092 aux_cap_helper__b9,r9,band_LOW_PAIRED__condX1
        + 4.4504289328 aux_cap_helper__b9,r9,band_LOW_PAIRED__condX2
        - 1.2715511236 aux_cap_helper__b9,r9,band_LOW_PAIRED__Z2  = 0
 c1300: aux_cap_helper__b6,r12,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b6,r12,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b6,r12,band_HIGH_PAIRED__Z2  = 1
 c1301: aux_cap_b9,r9,band_LOW_PAIRED + aux_cap_b9,r9,band_HIGH_PAIRED
        + aux_cap_b9,r9,band_UNPAIRED - aux_c_b9,r9  = 0
 c1302: aux_cap_helper__b5,r13,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b5,r13,band_UNPAIRED__Z2 <= 0
 c1303: aux_cap_helper__b0,r3,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b0,r3,band_HIGH_PAIRED__Z1 <= 0
 c1304: 0.888446608 aux_cap_helper__b5,r4,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b5,r4,band_UNPAIRED__condX1
        - aux_cap_b5,r4,band_UNPAIRED
        - 0.253841888 aux_cap_helper__b5,r4,band_UNPAIRED__Z2  = 0
 c1305: - aux_cap_b4,r11,band_HIGH_PAIRED
        + 2.8394335959 aux_cap_helper__b4,r11,band_HIGH_PAIRED__condX2
        + 1.8253501688 aux_cap_helper__b4,r11,band_HIGH_PAIRED__condX3
        + 2.0281668543 aux_cap_helper__b4,r11,band_HIGH_PAIRED__condX1
        - 0.8112667416 aux_cap_helper__b4,r11,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b4,r11,band_HIGH_PAIRED__Z3  = 0
 c1306: - sv_function_helpervar_b2,r8__condX2
        + 0.15845308692762 sv_function_helpervar_b2,r8__Z2 <= 0
 c1307: - 2 aux_cap_helper__b5,r7,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b5,r7,band_HIGH_PAIRED__condX2 <= 0
 c1308: aux_cap_helper__b3,r6,band_UNPAIRED__condX1
        - aux_cap_helper__b3,r6,band_UNPAIRED__Z1 <= 0
 c1309: - 2 aux_cap_helper__b8,r4,band_LOW_PAIRED__Z2
        + aux_cap_helper__b8,r4,band_LOW_PAIRED__condX2 <= 0
 c1310: - aux_Omega_b3,r9 + 145.86571128125 aux_sv_b3,r9  = 0
 c1311: aux_cap_helper__b3,r5,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b3,r5,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b3,r5,band_HIGH_PAIRED__condX3
        - X__b3,r5,band_HIGH_PAIRED  = 0
 c1312: - aux_sv_b2,r13 + 17.4163604891646 sv_function_helpervar_b2,r13__Z2
        + 36.1429684982152 sv_function_helpervar_b2,r13__Z3
        + 114.716693926 sv_function_helpervar_b2,r13__condX1
        + 4.8017605689 sv_function_helpervar_b2,r13__condX2
        + 1.9677778417 sv_function_helpervar_b2,r13__condX3  = 0
 c1313: sv_function_helpervar_b5,r11__Z3 + sv_function_helpervar_b5,r11__Z2
        + sv_function_helpervar_b5,r11__Z1  = 1
 c1314: - aux_cap_helper__b8,r10,band_HIGH_PAIRED__condX1 <= 0
 c1315: - aux_cap_b5,r13,band_LOW_PAIRED
        + 4.4504289328 aux_cap_helper__b5,r13,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b5,r13,band_LOW_PAIRED__condX1
        - 1.2715511236 aux_cap_helper__b5,r13,band_LOW_PAIRED__Z2  = 0
 c1316: 312.664640256681 sv_function_helpervar_b6,r10__Z3
        + 142.43366837227 sv_function_helpervar_b6,r10__Z2 - aux_sv_b6,r10
        + 15.1725304141 sv_function_helpervar_b6,r10__condX3
        + 43.5296666815 sv_function_helpervar_b6,r10__condX2
        + 942.4308567218 sv_function_helpervar_b6,r10__condX1  = 0
 c1317: - aux_Omega_b6,r3 + 379.259003493388 aux_sv_b6,r3  = 0
 c1318: X__b8,r3,band_UNPAIRED + X__b0,r3,band_UNPAIRED
        + X__b4,r3,band_UNPAIRED + X__b9,r3,band_UNPAIRED
        + X__b5,r3,band_UNPAIRED + X__b6,r3,band_UNPAIRED
        + X__b3,r3,band_UNPAIRED + X__b1,r3,band_UNPAIRED
        + X__b2,r3,band_UNPAIRED + X__b7,r3,band_UNPAIRED <= 2
 c1319: aux_cap_helper__b6,r0,band_LOW_PAIRED__Z1
        + aux_cap_helper__b6,r0,band_LOW_PAIRED__Z2  = 1
 c1320: 0.158453086943648 sv_function_helpervar_b7,r8__Z2
        - sv_function_helpervar_b7,r8__condX2 <= 0
 c1321: aux_cap_helper__b1,r8,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b1,r8,band_HIGH_PAIRED__condX2 <= 0
 c1322: aux_cap_helper__b4,r1,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b4,r1,band_UNPAIRED__Z2 <= 0
 c1323: - 3 aux_cap_helper__b5,r1,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b5,r1,band_HIGH_PAIRED__condX3 <= 0
 c1324: aux_cap_b0,r4,band_UNPAIRED - aux_c_b0,r4
        + aux_cap_b0,r4,band_HIGH_PAIRED + aux_cap_b0,r4,band_LOW_PAIRED  = 0
 c1325: sv_function_helpervar_b0,r4__condX3
        + sv_function_helpervar_b0,r4__condX2
        + sv_function_helpervar_b0,r4__condX1 - aux_c_b0,r4  = 0
 c1326: - aux_cap_helper__b5,r11,band_LOW_PAIRED__condX1 <= 0
 c1327: aux_cap_helper__b7,r8,band_LOW_PAIRED__Z2
        - aux_cap_helper__b7,r8,band_LOW_PAIRED__condX2 <= 0
 c1328: - aux_cap_helper__b1,r8,band_HIGH_PAIRED__condX1 <= 0
 c1329: 0.158453086892444 sv_function_helpervar_b2,r7__Z2
        - sv_function_helpervar_b2,r7__condX2 <= 0
 c1330: - 2 aux_cap_helper__b2,r13,band_LOW_PAIRED__Z2
        + aux_cap_helper__b2,r13,band_LOW_PAIRED__condX2 <= 0
 c1331: aux_cap_helper__b8,r9,band_UNPAIRED__Z2
        + aux_cap_helper__b8,r9,band_UNPAIRED__Z1  = 1
 c1332: aux_cap_helper__b1,r12,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b1,r12,band_UNPAIRED__Z2 <= 0
 c1333: sv_function_helpervar_b5,r3__Z2 + sv_function_helpervar_b5,r3__Z1
        + sv_function_helpervar_b5,r3__Z3  = 1
 c1334: X__b9,r0,band_HIGH_PAIRED + X__b9,r0,band_LOW_PAIRED
        - GlobalBidder_W_ir__b9,r0 + X__b9,r0,band_UNPAIRED >= 0
 c1335: X__b7,r5,band_UNPAIRED + X__b7,r5,band_HIGH_PAIRED
        - GlobalBidder_W_ir__b7,r5 + X__b7,r5,band_LOW_PAIRED >= 0
 c1336: aux_cap_helper__b5,r10,band_UNPAIRED__Z2
        + aux_cap_helper__b5,r10,band_UNPAIRED__Z1  = 1
 c1337: - aux_c_b5,r6 + sv_function_helpervar_b5,r6__condX1
        + sv_function_helpervar_b5,r6__condX2
        + sv_function_helpervar_b5,r6__condX3  = 0
 c1338: aux_cap_helper__b8,r10,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b8,r10,band_HIGH_PAIRED__condX2 <= 0
 c1339: - 0.158453086923784 sv_function_helpervar_b3,r2__Z1
        + sv_function_helpervar_b3,r2__condX1 <= 0
 c1340: sv_function_helpervar_b7,r0__Z3 + sv_function_helpervar_b7,r0__Z2
        + sv_function_helpervar_b7,r0__Z1  = 1
 c1341: - aux_Omega_b2,r5 + 341.412183777885 aux_sv_b2,r5  = 0
 c1342: - sv_function_helpervar_b4,r1__condX1 <= 0
 c1343: - 0.158453086904992 sv_function_helpervar_b7,r10__Z1
        + sv_function_helpervar_b7,r10__condX1 <= 0
 c1344: - 2 aux_cap_helper__b7,r0,band_UNPAIRED__Z2
        + aux_cap_helper__b7,r0,band_UNPAIRED__condX2 <= 0
 c1345: aux_cap_b1,r8,band_UNPAIRED + aux_cap_b1,r8,band_HIGH_PAIRED
        + aux_cap_b1,r8,band_LOW_PAIRED - aux_c_b1,r8  = 0
 c1346: - aux_cap_helper__b2,r9,band_UNPAIRED__condX1 <= 0
 c1347: - aux_cap_helper__b5,r7,band_LOW_PAIRED__condX2
        + aux_cap_helper__b5,r7,band_LOW_PAIRED__Z2 <= 0
 c1348: aux_cap_helper__b3,r7,band_UNPAIRED__condX1
        + aux_cap_helper__b3,r7,band_UNPAIRED__condX2 - X__b3,r7,band_UNPAIRED
         = 0
 c1349: aux_cap_helper__b4,r2,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b4,r2,band_HIGH_PAIRED__condX2 <= 0
 c1350: - 15.8453086887446 sv_function_helpervar_b0,r4__Z3
        + sv_function_helpervar_b0,r4__condX3 <= 0
 c1351: - aux_cap_b7,r2,band_LOW_PAIRED
        + 4.4504289328 aux_cap_helper__b7,r2,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b7,r2,band_LOW_PAIRED__condX1
        - 1.2715511236 aux_cap_helper__b7,r2,band_LOW_PAIRED__Z2  = 0
 c1352: aux_cap_helper__b1,r3,band_UNPAIRED__Z1
        + aux_cap_helper__b1,r3,band_UNPAIRED__Z2  = 1
 c1353: - aux_c_b6,r3 + sv_function_helpervar_b6,r3__condX2
        + sv_function_helpervar_b6,r3__condX3
        + sv_function_helpervar_b6,r3__condX1  = 0
 c1354: aux_cap_helper__b0,r9,band_LOW_PAIRED__condX1
        - aux_cap_helper__b0,r9,band_LOW_PAIRED__Z1 <= 0
 c1355: aux_cap_helper__b8,r5,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b8,r5,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b8,r5,band_HIGH_PAIRED__Z3  = 1
 c1356: - aux_cap_helper__b5,r7,band_LOW_PAIRED__condX1 <= 0
 c1357: - aux_c_b8,r5 + aux_cap_b8,r5,band_HIGH_PAIRED
        + aux_cap_b8,r5,band_UNPAIRED + aux_cap_b8,r5,band_LOW_PAIRED  = 0
 c1358: 4.4504289328 aux_cap_helper__b8,r12,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b8,r12,band_LOW_PAIRED__condX1
        - 1.2715511236 aux_cap_helper__b8,r12,band_LOW_PAIRED__Z2
        - aux_cap_b8,r12,band_LOW_PAIRED  = 0
 c1359: - aux_cap_helper__b4,r2,band_HIGH_PAIRED__condX1 <= 0
 c1360: X__b5,r8,band_UNPAIRED + X__b7,r8,band_UNPAIRED
        + X__b2,r8,band_UNPAIRED + X__b6,r8,band_UNPAIRED
        + X__b1,r8,band_UNPAIRED + X__b0,r8,band_UNPAIRED
        + X__b9,r8,band_UNPAIRED + X__b3,r8,band_UNPAIRED
        + X__b4,r8,band_UNPAIRED + X__b8,r8,band_UNPAIRED <= 2
 c1361: aux_cap_helper__b6,r6,band_LOW_PAIRED__condX2
        + aux_cap_helper__b6,r6,band_LOW_PAIRED__condX1
        - X__b6,r6,band_LOW_PAIRED  = 0
 c1362: aux_cap_helper__b1,r10,band_LOW_PAIRED__Z2
        - aux_cap_helper__b1,r10,band_LOW_PAIRED__condX2 <= 0
 c1363: aux_cap_helper__b1,r12,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b1,r12,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b1,r12,band_HIGH_PAIRED__Z2  = 1
 c1364: - 1.2715511236 aux_cap_helper__b7,r9,band_LOW_PAIRED__Z2
        - aux_cap_b7,r9,band_LOW_PAIRED
        + 4.4504289328 aux_cap_helper__b7,r9,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b7,r9,band_LOW_PAIRED__condX1  = 0
 c1365: - aux_cap_helper__b6,r12,band_UNPAIRED__Z1
        + aux_cap_helper__b6,r12,band_UNPAIRED__condX1 <= 0
 c1366: aux_cap_helper__b6,r13,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b6,r13,band_HIGH_PAIRED__Z1 <= 0
 c1367: - X__b9,r12,band_UNPAIRED
        + aux_cap_helper__b9,r12,band_UNPAIRED__condX1
        + aux_cap_helper__b9,r12,band_UNPAIRED__condX2  = 0
 c1368: 6.01751054263764 sv_function_helpervar_b6,r8__Z3
        - sv_function_helpervar_b6,r8__condX3 <= 0
 c1369: - aux_cap_helper__b7,r1,band_HIGH_PAIRED__condX1 <= 0
 c1370: sv_function_helpervar_b7,r2__condX3
        - 15.8453086887446 sv_function_helpervar_b7,r2__Z3 <= 0
 c1371: aux_cap_helper__b7,r4,band_LOW_PAIRED__Z1
        + aux_cap_helper__b7,r4,band_LOW_PAIRED__Z2  = 1
 c1372: 6.36037231620095 sv_function_helpervar_b0,r10__Z3
        - sv_function_helpervar_b0,r10__condX3 <= 0
 c1373: X__b8,r9,band_LOW_PAIRED + X__b8,r9,band_HIGH_PAIRED
        + X__b8,r9,band_UNPAIRED - GlobalBidder_W_ir__b8,r9 >= 0
 c1374: - aux_cap_helper__b9,r2,band_UNPAIRED__condX1 <= 0
 c1375: aux_cap_b2,r1,band_UNPAIRED - aux_c_b2,r1
        + aux_cap_b2,r1,band_LOW_PAIRED + aux_cap_b2,r1,band_HIGH_PAIRED  = 0
 c1376: - aux_cap_b9,r11,band_LOW_PAIRED
        + 3.1788778092 aux_cap_helper__b9,r11,band_LOW_PAIRED__condX1
        + 4.4504289328 aux_cap_helper__b9,r11,band_LOW_PAIRED__condX2
        - 1.2715511236 aux_cap_helper__b9,r11,band_LOW_PAIRED__Z2  = 0
 c1377: aux_cap_helper__b6,r10,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b6,r10,band_HIGH_PAIRED__Z3 <= 0
 c1378: 0.42045 aux_Omega_b6,r10 + aux_Omega_b6,r11 + 0.42045 aux_Omega_b6,r12
        + 0.42045 aux_Omega_b6,r13 - v__6 + 0.01314 aux_Omega_b6,r3
        + 0.01314 aux_Omega_b6,r4 + 0.03125 aux_Omega_b6,r5
        + 0.03125 aux_Omega_b6,r6 + 0.03125 aux_Omega_b6,r7
        + 0.07433 aux_Omega_b6,r8 + 0.17678 aux_Omega_b6,r9
        + 0.01314 aux_Omega_b6,r0 + 0.00552 aux_Omega_b6,r1
        + 0.00552 aux_Omega_b6,r2  = 0
 c1379: - 1.2715511236 aux_cap_helper__b5,r1,band_LOW_PAIRED__Z2
        - aux_cap_b5,r1,band_LOW_PAIRED
        + 4.4504289328 aux_cap_helper__b5,r1,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b5,r1,band_LOW_PAIRED__condX1  = 0
 c1380: 0.0102040816326531 X__b8,r11,band_LOW_PAIRED
        + 0.0102040816326531 X__b8,r11,band_HIGH_PAIRED
        - GlobalBidder_W_ir__b8,r11
        + 0.0102040816326531 X__b8,r11,band_UNPAIRED <= 0
 c1381: - aux_cap_helper__b3,r6,band_LOW_PAIRED__condX1 <= 0
 c1382: aux_cap_helper__b5,r11,band_LOW_PAIRED__Z2
        - aux_cap_helper__b5,r11,band_LOW_PAIRED__condX2 <= 0
 c1383: - X__b4,r12,band_HIGH_PAIRED
        + aux_cap_helper__b4,r12,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b4,r12,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b4,r12,band_HIGH_PAIRED__condX1  = 0
 c1384: aux_cap_helper__b3,r2,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b3,r2,band_HIGH_PAIRED__Z2 <= 0
 c1385: sv_function_helpervar_b8,r10__condX3
        - 15.8453086887446 sv_function_helpervar_b8,r10__Z3 <= 0
 c1386: - X__b2,r9,band_LOW_PAIRED
        + aux_cap_helper__b2,r9,band_LOW_PAIRED__condX1
        + aux_cap_helper__b2,r9,band_LOW_PAIRED__condX2  = 0
 c1387: aux_cap_helper__b7,r13,band_LOW_PAIRED__condX1
        - aux_cap_helper__b7,r13,band_LOW_PAIRED__Z1 <= 0
 c1388: aux_cap_helper__b3,r11,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b3,r11,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b3,r11,band_HIGH_PAIRED__condX1
        - X__b3,r11,band_HIGH_PAIRED  = 0
 c1389: - 2 aux_cap_helper__b7,r5,band_UNPAIRED__Z2
        + aux_cap_helper__b7,r5,band_UNPAIRED__condX2 <= 0
 c1390: 7.00044505459004 sv_function_helpervar_b7,r0__Z3
        - sv_function_helpervar_b7,r0__condX3 <= 0
 c1391: aux_cap_helper__b0,r2,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b0,r2,band_HIGH_PAIRED__Z2 <= 0
 c1392: sv_function_helpervar_b0,r7__Z1 + sv_function_helpervar_b0,r7__Z2
        + sv_function_helpervar_b0,r7__Z3  = 1
 c1393: 2 aux_cap_helper__b3,r12,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b3,r12,band_HIGH_PAIRED__condX3 <= 0
 c1394: aux_cap_helper__b7,r1,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b7,r1,band_HIGH_PAIRED__condX2 <= 0
 c1395: - sv_function_helpervar_b8,r12__condX2
        + 0.158453086927029 sv_function_helpervar_b8,r12__Z2 <= 0
 c1396: 6.19049285847949 sv_function_helpervar_b9,r9__Z3
        - sv_function_helpervar_b9,r9__condX3 <= 0
 c1397: aux_cap_helper__b2,r11,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b2,r11,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b2,r11,band_HIGH_PAIRED__Z1  = 1
 c1398: GlobalBidder_PSI__i7,k0 - 20668026.8748422 GlobalBidder_hat{W}__i7,k0
        <= 0
 c1399: - 2 aux_cap_helper__b6,r0,band_UNPAIRED__Z2
        + aux_cap_helper__b6,r0,band_UNPAIRED__condX2 <= 0
 c1400: aux_cap_helper__b9,r2,band_UNPAIRED__Z2
        - aux_cap_helper__b9,r2,band_UNPAIRED__condX2 <= 0
 c1401: - X__b4,r6,band_LOW_PAIRED
        + aux_cap_helper__b4,r6,band_LOW_PAIRED__condX2
        + aux_cap_helper__b4,r6,band_LOW_PAIRED__condX1  = 0
 c1402: sv_function_helpervar_b1,r3__Z2 + sv_function_helpervar_b1,r3__Z1
        + sv_function_helpervar_b1,r3__Z3  = 1
 c1403: aux_cap_b5,r4,band_LOW_PAIRED - aux_c_b5,r4
        + aux_cap_b5,r4,band_UNPAIRED + aux_cap_b5,r4,band_HIGH_PAIRED  = 0
 c1404: - aux_c_b0,r7 + sv_function_helpervar_b0,r7__condX1
        + sv_function_helpervar_b0,r7__condX2
        + sv_function_helpervar_b0,r7__condX3  = 0
 c1405: aux_cap_helper__b3,r6,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b3,r6,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b3,r6,band_HIGH_PAIRED__Z3  = 1
 c1406: aux_cap_helper__b5,r1,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b5,r1,band_UNPAIRED__Z2 <= 0
 c1407: sv_function_helpervar_b1,r12__condX2
        + sv_function_helpervar_b1,r12__condX1
        + sv_function_helpervar_b1,r12__condX3 - aux_c_b1,r12  = 0
 c1408: aux_cap_helper__b1,r7,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b1,r7,band_LOW_PAIRED__Z2 <= 0
 c1409: - 7.12654647039803 sv_function_helpervar_b9,r8__Z2
        + sv_function_helpervar_b9,r8__condX2 <= 0
 c1410: aux_cap_helper__b9,r13,band_UNPAIRED__condX1
        - aux_cap_helper__b9,r13,band_UNPAIRED__Z1 <= 0
 c1411: aux_cap_helper__b3,r6,band_LOW_PAIRED__Z2
        - aux_cap_helper__b3,r6,band_LOW_PAIRED__condX2 <= 0
 c1412: aux_cap_helper__b4,r3,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b4,r3,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b4,r3,band_HIGH_PAIRED__Z1  = 1
 c1413: - aux_cap_helper__b4,r1,band_HIGH_PAIRED__condX1 <= 0
 c1414: - sv_function_helpervar_b8,r4__condX1 <= 0
 c1415: 42.753270283 sv_function_helpervar_b7,r7__condX2
        + 1255.6694400889 sv_function_helpervar_b7,r7__condX1
        + 25.6458021029 sv_function_helpervar_b7,r7__condX3
        + 192.190311254669 sv_function_helpervar_b7,r7__Z2
        + 330.540641354152 sv_function_helpervar_b7,r7__Z3 - aux_sv_b7,r7  = 0
 c1416: aux_cap_helper__b8,r3,band_LOW_PAIRED__condX1
        + aux_cap_helper__b8,r3,band_LOW_PAIRED__condX2
        - X__b8,r3,band_LOW_PAIRED  = 0
 c1417: - 0.158453086903266 sv_function_helpervar_b8,r13__Z1
        + sv_function_helpervar_b8,r13__condX1 <= 0
 c1418: - 0.158453086912404 sv_function_helpervar_b9,r12__Z1
        + sv_function_helpervar_b9,r12__condX1 <= 0
 c1419: - aux_cap_helper__b2,r10,band_HIGH_PAIRED__condX1 <= 0
 c1420: - aux_Omega_b1,r0 + 160.189275218141 aux_sv_b1,r0  = 0
 c1421: - aux_cap_helper__b0,r11,band_LOW_PAIRED__condX2
        + aux_cap_helper__b0,r11,band_LOW_PAIRED__Z2 <= 0
 c1422: aux_cap_helper__b1,r8,band_LOW_PAIRED__Z2
        - aux_cap_helper__b1,r8,band_LOW_PAIRED__condX2 <= 0
 c1423: - aux_c_b7,r6 + sv_function_helpervar_b7,r6__condX3
        + sv_function_helpervar_b7,r6__condX2
        + sv_function_helpervar_b7,r6__condX1  = 0
 c1424: 3.1788778092 aux_cap_helper__b4,r13,band_LOW_PAIRED__condX1
        + 4.4504289328 aux_cap_helper__b4,r13,band_LOW_PAIRED__condX2
        - 1.2715511236 aux_cap_helper__b4,r13,band_LOW_PAIRED__Z2
        - aux_cap_b4,r13,band_LOW_PAIRED  = 0
 c1425: sv_function_helpervar_b8,r6__condX1
        - 0.158453086895541 sv_function_helpervar_b8,r6__Z1 <= 0
 c1426: aux_cap_helper__b5,r2,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b5,r2,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b5,r2,band_HIGH_PAIRED__condX2
        - X__b5,r2,band_HIGH_PAIRED  = 0
 c1427: - 5.79372069706504 sv_function_helpervar_b0,r0__Z2
        + sv_function_helpervar_b0,r0__condX2 <= 0
 c1428: sv_function_helpervar_b6,r11__condX1
        - 0.15845308690825 sv_function_helpervar_b6,r11__Z1 <= 0
 c1429: 6.59791083704279 sv_function_helpervar_b9,r1__Z3
        - sv_function_helpervar_b9,r1__condX3 <= 0
 c1430: aux_cap_helper__b2,r3,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b2,r3,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b2,r3,band_HIGH_PAIRED__Z2  = 1
 c1431: - aux_cap_helper__b0,r11,band_LOW_PAIRED__condX1 <= 0
 c1432: - aux_cap_helper__b1,r8,band_LOW_PAIRED__condX1 <= 0
 c1433: GlobalBidder_W_i_9 + 14 GlobalBidder_hat{W}__i9,k0 <= 28
 c1434: 0.158453086984489 sv_function_helpervar_b3,r1__Z2
        - sv_function_helpervar_b3,r1__condX2 <= 0
 c1435: aux_cap_helper__b7,r0,band_UNPAIRED__Z1
        + aux_cap_helper__b7,r0,band_UNPAIRED__Z2  = 1
 c1436: 0.63460472 aux_cap_helper__b7,r1,band_UNPAIRED__condX1
        + 0.888446608 aux_cap_helper__b7,r1,band_UNPAIRED__condX2
        - 0.253841888 aux_cap_helper__b7,r1,band_UNPAIRED__Z2
        - aux_cap_b7,r1,band_UNPAIRED  = 0
 c1437: - 2 aux_cap_helper__b8,r7,band_LOW_PAIRED__Z2
        + aux_cap_helper__b8,r7,band_LOW_PAIRED__condX2 <= 0
 c1438: aux_cap_helper__b4,r7,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b4,r7,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b4,r7,band_HIGH_PAIRED__condX2
        - X__b4,r7,band_HIGH_PAIRED  = 0
 c1439: aux_cap_helper__b6,r10,band_LOW_PAIRED__Z2
        - aux_cap_helper__b6,r10,band_LOW_PAIRED__condX2 <= 0
 c1440: - 2 aux_cap_helper__b6,r3,band_LOW_PAIRED__Z2
        + aux_cap_helper__b6,r3,band_LOW_PAIRED__condX2 <= 0
 c1441: aux_cap_helper__b4,r4,band_UNPAIRED__Z1
        + aux_cap_helper__b4,r4,band_UNPAIRED__Z2  = 1
 c1442: 3.1788778092 aux_cap_helper__b3,r8,band_LOW_PAIRED__condX1
        + 4.4504289328 aux_cap_helper__b3,r8,band_LOW_PAIRED__condX2
        - aux_cap_b3,r8,band_LOW_PAIRED
        - 1.2715511236 aux_cap_helper__b3,r8,band_LOW_PAIRED__Z2  = 0
 c1443: - X__b4,r11,band_LOW_PAIRED
        + aux_cap_helper__b4,r11,band_LOW_PAIRED__condX1
        + aux_cap_helper__b4,r11,band_LOW_PAIRED__condX2  = 0
 c1444: - aux_cap_helper__b9,r9,band_LOW_PAIRED__condX2
        + aux_cap_helper__b9,r9,band_LOW_PAIRED__Z2 <= 0
 c1445: - aux_cap_helper__b4,r1,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b4,r1,band_HIGH_PAIRED__Z2 <= 0
 c1446: aux_cap_helper__b8,r2,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b8,r2,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b8,r2,band_HIGH_PAIRED__condX1
        - X__b8,r2,band_HIGH_PAIRED  = 0
 c1447: - aux_cap_helper__b2,r10,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b2,r10,band_HIGH_PAIRED__Z2 <= 0
 c1448: 2 aux_cap_helper__b9,r5,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b9,r5,band_HIGH_PAIRED__condX3 <= 0
 c1449: aux_cap_helper__b2,r9,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b2,r9,band_HIGH_PAIRED__Z1 <= 0
 c1450: - aux_cap_helper__b7,r5,band_LOW_PAIRED__condX2
        + aux_cap_helper__b7,r5,band_LOW_PAIRED__Z2 <= 0
 c1451: 141.689810360043 sv_function_helpervar_b6,r4__Z2
        + 325.803057187971 sv_function_helpervar_b6,r4__Z3
        + 942.4308566974 sv_function_helpervar_b6,r4__condX1
        + 48.2241666538 sv_function_helpervar_b6,r4__condX2
        + 14.3433627917 sv_function_helpervar_b6,r4__condX3 - aux_sv_b6,r4  = 0
 c1452: 0.888446608 aux_cap_helper__b8,r6,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b8,r6,band_UNPAIRED__condX1
        - aux_cap_b8,r6,band_UNPAIRED
        - 0.253841888 aux_cap_helper__b8,r6,band_UNPAIRED__Z2  = 0
 c1453: aux_cap_helper__b9,r0,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b9,r0,band_UNPAIRED__Z2 <= 0
 c1454: aux_cap_helper__b5,r6,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b5,r6,band_UNPAIRED__Z2 <= 0
 c1455: - aux_cap_helper__b7,r5,band_LOW_PAIRED__condX1 <= 0
 c1456: - 15.8453086887446 sv_function_helpervar_b1,r8__Z3
        + sv_function_helpervar_b1,r8__condX3 <= 0
 c1457: aux_cap_helper__b2,r2,band_LOW_PAIRED__Z2
        + aux_cap_helper__b2,r2,band_LOW_PAIRED__Z1  = 1
 c1458: - aux_cap_helper__b2,r1,band_UNPAIRED__Z1
        + aux_cap_helper__b2,r1,band_UNPAIRED__condX1 <= 0
 c1459: aux_cap_b5,r0,band_HIGH_PAIRED + aux_cap_b5,r0,band_UNPAIRED
        - aux_c_b5,r0 + aux_cap_b5,r0,band_LOW_PAIRED  = 0
 c1460: - sv_function_helpervar_b7,r1__condX1 <= 0
 c1461: 0.1584530869274 sv_function_helpervar_b7,r1__Z2
        - sv_function_helpervar_b7,r1__condX2 <= 0
 c1462: - 2 aux_cap_helper__b0,r7,band_UNPAIRED__Z2
        + aux_cap_helper__b0,r7,band_UNPAIRED__condX2 <= 0
 c1463: aux_cap_helper__b0,r10,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b0,r10,band_HIGH_PAIRED__Z1 <= 0
 c1464: aux_cap_helper__b8,r0,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b8,r0,band_HIGH_PAIRED__Z2 <= 0
 c1465: 4.4504289328 aux_cap_helper__b9,r12,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b9,r12,band_LOW_PAIRED__condX1
        - 1.2715511236 aux_cap_helper__b9,r12,band_LOW_PAIRED__Z2
        - aux_cap_b9,r12,band_LOW_PAIRED  = 0
 c1466: - X__b6,r12,band_UNPAIRED
        + aux_cap_helper__b6,r12,band_UNPAIRED__condX2
        + aux_cap_helper__b6,r12,band_UNPAIRED__condX1  = 0
 c1467: - sv_function_helpervar_b0,r3__condX3
        + 6.31602055756089 sv_function_helpervar_b0,r3__Z3 <= 0
 c1468: sv_function_helpervar_b6,r9__condX3
        - 15.8453086887446 sv_function_helpervar_b6,r9__Z3 <= 0
 c1469: aux_cap_helper__b7,r7,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b7,r7,band_HIGH_PAIRED__Z3 <= 0
 c1470: - aux_Omega_b1,r2 + 123.452969389444 aux_sv_b1,r2  = 0
 c1471: - X__b1,r9,band_UNPAIRED + aux_cap_helper__b1,r9,band_UNPAIRED__condX2
        + aux_cap_helper__b1,r9,band_UNPAIRED__condX1  = 0
 c1472: - aux_cap_helper__b6,r8,band_HIGH_PAIRED__condX1 <= 0
 c1473: sv_function_helpervar_b6,r13__condX2
        - 5.8935145383666 sv_function_helpervar_b6,r13__Z2 <= 0
 c1474: - sv_function_helpervar_b2,r0__condX1 <= 0
 c1475: - aux_cap_helper__b7,r3,band_UNPAIRED__condX1 <= 0
 c1476: aux_cap_helper__b7,r11,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b7,r11,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b7,r11,band_HIGH_PAIRED__Z3  = 1
 c1477: aux_cap_helper__b2,r6,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b2,r6,band_UNPAIRED__Z2 <= 0
 c1478: - aux_cap_helper__b5,r9,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b5,r9,band_HIGH_PAIRED__condX1 <= 0
 c1479: aux_cap_helper__b8,r12,band_LOW_PAIRED__condX1
        - aux_cap_helper__b8,r12,band_LOW_PAIRED__Z1 <= 0
 c1480: aux_cap_helper__b7,r2,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b7,r2,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b7,r2,band_HIGH_PAIRED__Z2  = 1
 c1481: - 3 aux_cap_helper__b4,r7,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b4,r7,band_HIGH_PAIRED__condX3 <= 0
 c1482: - X__b4,r0,band_HIGH_PAIRED
        + aux_cap_helper__b4,r0,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b4,r0,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b4,r0,band_HIGH_PAIRED__condX1  = 0
 c1483: aux_cap_helper__b0,r13,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b0,r13,band_HIGH_PAIRED__Z1 <= 0
 c1484: 18.2840050378 sv_function_helpervar_b3,r10__condX3
        + 1152.1877366553 sv_function_helpervar_b3,r10__condX1
        + 54.5518867557 sv_function_helpervar_b3,r10__condX2
        + 386.460975953379 sv_function_helpervar_b3,r10__Z3
        + 173.923788705202 sv_function_helpervar_b3,r10__Z2 - aux_sv_b3,r10
         = 0
 c1485: aux_cap_helper__b8,r3,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b8,r3,band_HIGH_PAIRED__Z2 <= 0
 c1486: aux_cap_helper__b5,r6,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b5,r6,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b5,r6,band_HIGH_PAIRED__Z3  = 1
 c1487: aux_cap_helper__b0,r2,band_UNPAIRED__condX1
        - aux_cap_helper__b0,r2,band_UNPAIRED__Z1 <= 0
 c1488: - 0.253841888 aux_cap_helper__b2,r8,band_UNPAIRED__Z2
        - aux_cap_b2,r8,band_UNPAIRED
        + 0.888446608 aux_cap_helper__b2,r8,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b2,r8,band_UNPAIRED__condX1  = 0
 c1489: aux_cap_helper__b5,r12,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b5,r12,band_HIGH_PAIRED__Z1 <= 0
 c1490: aux_cap_helper__b9,r7,band_UNPAIRED__Z2
        - aux_cap_helper__b9,r7,band_UNPAIRED__condX2 <= 0
 c1491: - sv_function_helpervar_b8,r4__condX3
        + 7.85656987996407 sv_function_helpervar_b8,r4__Z3 <= 0
 c1492: 883.4028073458 sv_function_helpervar_b4,r9__condX1
        + 14.1544281469 sv_function_helpervar_b4,r9__condX3
        + 41.1349859488 sv_function_helpervar_b4,r9__condX2 - aux_sv_b4,r9
        + 294.155390258087 sv_function_helpervar_b4,r9__Z3
        + 133.459936354539 sv_function_helpervar_b4,r9__Z2  = 0
 c1493: aux_cap_helper__b0,r5,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b0,r5,band_HIGH_PAIRED__Z2 <= 0
 c1494: sv_function_helpervar_b1,r11__condX3
        - 15.8453086887446 sv_function_helpervar_b1,r11__Z3 <= 0
 c1495: 2 aux_cap_helper__b1,r7,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b1,r7,band_HIGH_PAIRED__condX3 <= 0
 c1496: X__b8,r7,band_LOW_PAIRED + X__b8,r7,band_UNPAIRED
        - GlobalBidder_W_ir__b8,r7 + X__b8,r7,band_HIGH_PAIRED >= 0
 c1497: - 3 aux_cap_helper__b7,r4,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b7,r4,band_HIGH_PAIRED__condX3 <= 0
 c1498: aux_cap_helper__b9,r8,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b9,r8,band_HIGH_PAIRED__condX2 <= 0
 c1499: - 0.253841888 aux_cap_helper__b0,r5,band_UNPAIRED__Z2
        + 0.63460472 aux_cap_helper__b0,r5,band_UNPAIRED__condX1
        + 0.888446608 aux_cap_helper__b0,r5,band_UNPAIRED__condX2
        - aux_cap_b0,r5,band_UNPAIRED  = 0
 c1500: aux_cap_helper__b2,r9,band_LOW_PAIRED__Z2
        + aux_cap_helper__b2,r9,band_LOW_PAIRED__Z1  = 1
 c1501: - sv_function_helpervar_b1,r10__condX1 <= 0
 c1502: aux_cap_helper__b3,r10,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b3,r10,band_UNPAIRED__Z2 <= 0
 c1503: - aux_cap_helper__b9,r7,band_UNPAIRED__condX1 <= 0
 c1504: - X__b7,r0,band_HIGH_PAIRED
        + aux_cap_helper__b7,r0,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b7,r0,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b7,r0,band_HIGH_PAIRED__condX1  = 0
 c1505: aux_cap_b6,r12,band_HIGH_PAIRED + aux_cap_b6,r12,band_LOW_PAIRED
        + aux_cap_b6,r12,band_UNPAIRED - aux_c_b6,r12  = 0
 c1506: sv_function_helpervar_b2,r10__Z3 + sv_function_helpervar_b2,r10__Z2
        + sv_function_helpervar_b2,r10__Z1  = 1
 c1507: - aux_cap_helper__b9,r8,band_HIGH_PAIRED__condX1 <= 0
 c1508: aux_cap_helper__b6,r8,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b6,r8,band_HIGH_PAIRED__condX2 <= 0
 c1509: - 1.2715511236 aux_cap_helper__b1,r0,band_LOW_PAIRED__Z2
        + 3.1788778092 aux_cap_helper__b1,r0,band_LOW_PAIRED__condX1
        + 4.4504289328 aux_cap_helper__b1,r0,band_LOW_PAIRED__condX2
        - aux_cap_b1,r0,band_LOW_PAIRED  = 0
 c1510: aux_cap_helper__b7,r3,band_UNPAIRED__Z2
        - aux_cap_helper__b7,r3,band_UNPAIRED__condX2 <= 0
 c1511: 2 aux_cap_helper__b4,r4,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b4,r4,band_HIGH_PAIRED__condX3 <= 0
 c1512: aux_cap_helper__b4,r12,band_LOW_PAIRED__Z2
        + aux_cap_helper__b4,r12,band_LOW_PAIRED__Z1  = 1
 c1513: aux_cap_helper__b7,r1,band_LOW_PAIRED__condX1
        - aux_cap_helper__b7,r1,band_LOW_PAIRED__Z1 <= 0
 c1514: sv_function_helpervar_b6,r7__Z3 + sv_function_helpervar_b6,r7__Z2
        + sv_function_helpervar_b6,r7__Z1  = 1
 c1515: sv_function_helpervar_b6,r12__condX1
        - 0.158453086892631 sv_function_helpervar_b6,r12__Z1 <= 0
 c1516: 2.0281668543 aux_cap_helper__b0,r13,band_HIGH_PAIRED__condX1
        + 2.8394335959 aux_cap_helper__b0,r13,band_HIGH_PAIRED__condX2
        + 1.8253501688 aux_cap_helper__b0,r13,band_HIGH_PAIRED__condX3
        - 0.8112667416 aux_cap_helper__b0,r13,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b0,r13,band_HIGH_PAIRED__Z3
        - aux_cap_b0,r13,band_HIGH_PAIRED  = 0
 c1517: 2 aux_cap_helper__b1,r4,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b1,r4,band_HIGH_PAIRED__condX3 <= 0
 c1518: - aux_cap_helper__b9,r10,band_LOW_PAIRED__condX1 <= 0
 c1519: aux_cap_helper__b9,r9,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b9,r9,band_HIGH_PAIRED__Z3 <= 0
 c1520: - aux_cap_helper__b2,r13,band_UNPAIRED__condX1 <= 0
 c1521: 2 aux_cap_helper__b7,r3,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b7,r3,band_HIGH_PAIRED__condX3 <= 0
 c1522: aux_cap_helper__b0,r7,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b0,r7,band_HIGH_PAIRED__Z1 <= 0
 c1523: aux_cap_helper__b3,r6,band_LOW_PAIRED__Z1
        + aux_cap_helper__b3,r6,band_LOW_PAIRED__Z2  = 1
 c1524: - X__b5,r4,band_LOW_PAIRED
        + aux_cap_helper__b5,r4,band_LOW_PAIRED__condX2
        + aux_cap_helper__b5,r4,band_LOW_PAIRED__condX1  = 0
 c1525: - aux_cap_helper__b4,r4,band_UNPAIRED__condX1 <= 0
 c1526: aux_cap_helper__b7,r6,band_UNPAIRED__condX1
        - aux_cap_helper__b7,r6,band_UNPAIRED__Z1 <= 0
 c1527: - X__b0,r11,band_HIGH_PAIRED
        + aux_cap_helper__b0,r11,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b0,r11,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b0,r11,band_HIGH_PAIRED__condX1  = 0
 c1528: aux_cap_helper__b3,r13,band_UNPAIRED__condX2
        + aux_cap_helper__b3,r13,band_UNPAIRED__condX1
        - X__b3,r13,band_UNPAIRED  = 0
 c1529: sv_function_helpervar_b8,r7__condX1
        - 0.158453086983862 sv_function_helpervar_b8,r7__Z1 <= 0
 c1530: aux_cap_helper__b3,r8,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b3,r8,band_LOW_PAIRED__Z2 <= 0
 c1531: - aux_cap_helper__b8,r6,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b8,r6,band_HIGH_PAIRED__condX1 <= 0
 c1532: - aux_c_b2,r4 + sv_function_helpervar_b2,r4__condX3
        + sv_function_helpervar_b2,r4__condX1
        + sv_function_helpervar_b2,r4__condX2  = 0
 c1533: - aux_cap_helper__b8,r10,band_UNPAIRED__Z1
        + aux_cap_helper__b8,r10,band_UNPAIRED__condX1 <= 0
 c1534: - sv_function_helpervar_b7,r13__condX1 <= 0
 c1535: sv_function_helpervar_b9,r4__condX3
        - 15.8453086887446 sv_function_helpervar_b9,r4__Z3 <= 0
 c1536: - 2 aux_cap_helper__b0,r1,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b0,r1,band_HIGH_PAIRED__condX2 <= 0
 c1537: - 2 aux_cap_helper__b7,r9,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b7,r9,band_HIGH_PAIRED__condX2 <= 0
 c1538: aux_cap_helper__b1,r6,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b1,r6,band_UNPAIRED__Z2 <= 0
 c1539: X__b9,r1,band_UNPAIRED + X__b1,r1,band_UNPAIRED
        + X__b0,r1,band_UNPAIRED + X__b4,r1,band_UNPAIRED
        + X__b5,r1,band_UNPAIRED + X__b6,r1,band_UNPAIRED
        + X__b2,r1,band_UNPAIRED + X__b8,r1,band_UNPAIRED
        + X__b7,r1,band_UNPAIRED + X__b3,r1,band_UNPAIRED <= 2
 c1540: - aux_cap_b0,r9,band_LOW_PAIRED
        + 3.1788778092 aux_cap_helper__b0,r9,band_LOW_PAIRED__condX1
        + 4.4504289328 aux_cap_helper__b0,r9,band_LOW_PAIRED__condX2
        - 1.2715511236 aux_cap_helper__b0,r9,band_LOW_PAIRED__Z2  = 0
 c1541: aux_cap_helper__b1,r2,band_UNPAIRED__condX1
        - aux_cap_helper__b1,r2,band_UNPAIRED__Z1 <= 0
 c1542: - aux_cap_b2,r3,band_LOW_PAIRED
        - 1.2715511236 aux_cap_helper__b2,r3,band_LOW_PAIRED__Z2
        + 3.1788778092 aux_cap_helper__b2,r3,band_LOW_PAIRED__condX1
        + 4.4504289328 aux_cap_helper__b2,r3,band_LOW_PAIRED__condX2  = 0
 c1543: - X__b4,r2,band_UNPAIRED + aux_cap_helper__b4,r2,band_UNPAIRED__condX1
        + aux_cap_helper__b4,r2,band_UNPAIRED__condX2  = 0
 c1544: - aux_cap_helper__b2,r13,band_UNPAIRED__condX2
        + aux_cap_helper__b2,r13,band_UNPAIRED__Z2 <= 0
 c1545: aux_cap_helper__b6,r9,band_LOW_PAIRED__Z2
        + aux_cap_helper__b6,r9,band_LOW_PAIRED__Z1  = 1
 c1546: - aux_cap_helper__b9,r10,band_LOW_PAIRED__condX2
        + aux_cap_helper__b9,r10,band_LOW_PAIRED__Z2 <= 0
 c1547: sv_function_helpervar_b6,r2__condX2
        - 6.03603404051893 sv_function_helpervar_b6,r2__Z2 <= 0
 c1548: aux_cap_helper__b7,r10,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b7,r10,band_HIGH_PAIRED__Z3 <= 0
 c1549: - aux_cap_helper__b6,r6,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b6,r6,band_HIGH_PAIRED__Z3 <= 0
 c1550: - aux_cap_helper__b3,r3,band_LOW_PAIRED__condX1 <= 0
 c1551: - 20668026.8748422 GlobalBidder_hat{W}__i8,k2 + GlobalBidder_PSI__i8,k2
        <= 0
 c1552: - sv_function_helpervar_b1,r7__condX2
        + 0.158453086922884 sv_function_helpervar_b1,r7__Z2 <= 0
 c1553: - 0.158453086934085 sv_function_helpervar_b1,r10__Z1
        + sv_function_helpervar_b1,r10__condX1 <= 0
 c1554: sv_function_helpervar_b1,r11__Z1 + sv_function_helpervar_b1,r11__Z3
        + sv_function_helpervar_b1,r11__Z2  = 1
 c1555: aux_cap_helper__b3,r5,band_UNPAIRED__Z1
        + aux_cap_helper__b3,r5,band_UNPAIRED__Z2  = 1
 c1556: aux_cap_helper__b2,r10,band_UNPAIRED__condX2
        + aux_cap_helper__b2,r10,band_UNPAIRED__condX1
        - X__b2,r10,band_UNPAIRED  = 0
 c1557: - aux_cap_helper__b9,r2,band_LOW_PAIRED__Z1
        + aux_cap_helper__b9,r2,band_LOW_PAIRED__condX1 <= 0
 c1558: aux_cap_helper__b2,r2,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b2,r2,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b2,r2,band_HIGH_PAIRED__condX1
        - X__b2,r2,band_HIGH_PAIRED  = 0
 c1559: - 6.22043635189836 sv_function_helpervar_b4,r11__Z2
        + sv_function_helpervar_b4,r11__condX2 <= 0
 c1560: - aux_cap_helper__b9,r7,band_LOW_PAIRED__Z1
        + aux_cap_helper__b9,r7,band_LOW_PAIRED__condX1 <= 0
 c1561: sv_function_helpervar_b3,r0__Z3 + sv_function_helpervar_b3,r0__Z2
        + sv_function_helpervar_b3,r0__Z1  = 1
 c1562: - 6.23552468667465 sv_function_helpervar_b1,r5__Z2
        + sv_function_helpervar_b1,r5__condX2 <= 0
 c1563: aux_cap_b6,r4,band_LOW_PAIRED - aux_c_b6,r4
        + aux_cap_b6,r4,band_HIGH_PAIRED + aux_cap_b6,r4,band_UNPAIRED  = 0
 c1564: X__b4,r3,band_HIGH_PAIRED + X__b0,r3,band_HIGH_PAIRED
        + X__b8,r3,band_HIGH_PAIRED + X__b1,r3,band_HIGH_PAIRED
        + X__b2,r3,band_HIGH_PAIRED + X__b9,r3,band_HIGH_PAIRED
        + X__b3,r3,band_HIGH_PAIRED + X__b7,r3,band_HIGH_PAIRED
        + X__b6,r3,band_HIGH_PAIRED + X__b5,r3,band_HIGH_PAIRED <= 3
 c1565: aux_cap_helper__b2,r0,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b2,r0,band_HIGH_PAIRED__condX2 <= 0
 c1566: 2 aux_cap_helper__b2,r12,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b2,r12,band_HIGH_PAIRED__condX3 <= 0
 c1567: - aux_cap_helper__b0,r10,band_UNPAIRED__condX1 <= 0
 c1568: - 20668026.8748422 GlobalBidder_hat{W}__i8,k2 - GlobalBidder_PSI__i8,k2
        + aux_Omega_b8,r12 + aux_Omega_b8,r13 + aux_Omega_b8,r10
        + aux_Omega_b8,r11 + aux_Omega_b8,r5 + aux_Omega_b8,r6
        + aux_Omega_b8,r7 + aux_Omega_b8,r8 + aux_Omega_b8,r9 + aux_Omega_b8,r0
        + aux_Omega_b8,r1 + aux_Omega_b8,r2 + aux_Omega_b8,r3 + aux_Omega_b8,r4
        >= -20668026.8748422
 c1569: aux_cap_helper__b1,r10,band_LOW_PAIRED__Z1
        + aux_cap_helper__b1,r10,band_LOW_PAIRED__Z2  = 1
 c1570: - aux_cap_helper__b5,r4,band_UNPAIRED__condX2
        + aux_cap_helper__b5,r4,band_UNPAIRED__Z2 <= 0
 c1571: 1293.7104729961 aux_sv_b8,r10 - aux_Omega_b8,r10  = 0
 c1572: aux_cap_helper__b6,r7,band_LOW_PAIRED__Z1
        + aux_cap_helper__b6,r7,band_LOW_PAIRED__Z2  = 1
 c1573: aux_cap_helper__b2,r1,band_UNPAIRED__Z2
        + aux_cap_helper__b2,r1,band_UNPAIRED__Z1  = 1
 c1574: aux_cap_helper__b6,r11,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b6,r11,band_HIGH_PAIRED__Z2 <= 0
 c1575: - aux_cap_helper__b4,r8,band_UNPAIRED__condX2
        + aux_cap_helper__b4,r8,band_UNPAIRED__Z2 <= 0
 c1576: - X__b1,r12,band_HIGH_PAIRED
        + aux_cap_helper__b1,r12,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b1,r12,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b1,r12,band_HIGH_PAIRED__condX1  = 0
 c1577: - aux_c_b5,r0 + sv_function_helpervar_b5,r0__condX1
        + sv_function_helpervar_b5,r0__condX3
        + sv_function_helpervar_b5,r0__condX2  = 0
 c1578: - aux_cap_helper__b2,r0,band_HIGH_PAIRED__condX1 <= 0
 c1579: 3.1788778092 aux_cap_helper__b3,r0,band_LOW_PAIRED__condX1
        + 4.4504289328 aux_cap_helper__b3,r0,band_LOW_PAIRED__condX2
        - 1.2715511236 aux_cap_helper__b3,r0,band_LOW_PAIRED__Z2
        - aux_cap_b3,r0,band_LOW_PAIRED  = 0
 c1580: aux_cap_helper__b3,r3,band_LOW_PAIRED__Z2
        - aux_cap_helper__b3,r3,band_LOW_PAIRED__condX2 <= 0
 c1581: - aux_cap_helper__b4,r8,band_UNPAIRED__condX1 <= 0
 c1582: aux_cap_helper__b1,r9,band_UNPAIRED__Z2
        + aux_cap_helper__b1,r9,band_UNPAIRED__Z1  = 1
 c1583: - aux_sv_b2,r10 + 1.8429932441 sv_function_helpervar_b2,r10__condX3
        + 5.317427372 sv_function_helpervar_b2,r10__condX2
        + 114.716693939 sv_function_helpervar_b2,r10__condX1
        + 38.1202189664535 sv_function_helpervar_b2,r10__Z3
        + 17.3346514924843 sv_function_helpervar_b2,r10__Z2  = 0
 c1584: - aux_cap_helper__b3,r13,band_LOW_PAIRED__condX2
        + aux_cap_helper__b3,r13,band_LOW_PAIRED__Z2 <= 0
 c1585: aux_cap_helper__b3,r4,band_LOW_PAIRED__Z1
        + aux_cap_helper__b3,r4,band_LOW_PAIRED__Z2  = 1
 c1586: sv_function_helpervar_b5,r0__condX1
        - 0.158453086925932 sv_function_helpervar_b5,r0__Z1 <= 0
 c1587: - aux_cap_helper__b3,r13,band_UNPAIRED__condX1 <= 0
 c1588: - aux_cap_b8,r0,band_LOW_PAIRED
        + 4.4504289328 aux_cap_helper__b8,r0,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b8,r0,band_LOW_PAIRED__condX1
        - 1.2715511236 aux_cap_helper__b8,r0,band_LOW_PAIRED__Z2  = 0
 c1589: sv_function_helpervar_b7,r3__Z1 + sv_function_helpervar_b7,r3__Z2
        + sv_function_helpervar_b7,r3__Z3  = 1
 c1590: - X__b4,r3,band_HIGH_PAIRED
        + aux_cap_helper__b4,r3,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b4,r3,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b4,r3,band_HIGH_PAIRED__condX1  = 0
 c1591: aux_cap_helper__b1,r4,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b1,r4,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b1,r4,band_HIGH_PAIRED__Z3  = 1
 c1592: 6.31923003595998 sv_function_helpervar_b3,r1__Z3
        - sv_function_helpervar_b3,r1__condX3 <= 0
 c1593: aux_cap_helper__b0,r1,band_LOW_PAIRED__condX1
        - aux_cap_helper__b0,r1,band_LOW_PAIRED__Z1 <= 0
 c1594: aux_cap_helper__b6,r7,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b6,r7,band_HIGH_PAIRED__condX2 <= 0
 c1595: aux_cap_helper__b2,r3,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b2,r3,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b2,r3,band_HIGH_PAIRED__condX1
        - X__b2,r3,band_HIGH_PAIRED  = 0
 c1596: aux_cap_helper__b5,r1,band_LOW_PAIRED__Z2
        - aux_cap_helper__b5,r1,band_LOW_PAIRED__condX2 <= 0
 c1597: - X__b9,r4,band_HIGH_PAIRED
        + aux_cap_helper__b9,r4,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b9,r4,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b9,r4,band_HIGH_PAIRED__condX3  = 0
 c1598: 29.2808020383483 aux_sv_b1,r13 - aux_Omega_b1,r13  = 0
 c1599: aux_cap_helper__b6,r5,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b6,r5,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b6,r5,band_HIGH_PAIRED__Z3  = 1
 c1600: aux_cap_helper__b7,r2,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b7,r2,band_HIGH_PAIRED__condX2 <= 0
 c1601: - aux_cap_b0,r3,band_UNPAIRED
        - 0.253841888 aux_cap_helper__b0,r3,band_UNPAIRED__Z2
        + 0.888446608 aux_cap_helper__b0,r3,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b0,r3,band_UNPAIRED__condX1  = 0
 c1602: - aux_cap_helper__b8,r11,band_LOW_PAIRED__condX2
        + aux_cap_helper__b8,r11,band_LOW_PAIRED__Z2 <= 0
 c1603: aux_cap_helper__b2,r2,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b2,r2,band_UNPAIRED__Z2 <= 0
 c1604: 0.158453086936884 sv_function_helpervar_b4,r1__Z2
        - sv_function_helpervar_b4,r1__condX2 <= 0
 c1605: - sv_function_helpervar_b9,r3__condX1 <= 0
 c1606: sv_function_helpervar_b7,r7__condX1
        - 0.158453086898351 sv_function_helpervar_b7,r7__Z1 <= 0
 c1607: aux_cap_b4,r12,band_UNPAIRED + aux_cap_b4,r12,band_HIGH_PAIRED
        + aux_cap_b4,r12,band_LOW_PAIRED - aux_c_b4,r12  = 0
 c1608: - X__b6,r13,band_UNPAIRED
        + aux_cap_helper__b6,r13,band_UNPAIRED__condX1
        + aux_cap_helper__b6,r13,band_UNPAIRED__condX2  = 0
 c1609: - 5.40314922344626 sv_function_helpervar_b3,r0__Z2
        + sv_function_helpervar_b3,r0__condX2 <= 0
 c1610: - aux_cap_helper__b5,r1,band_LOW_PAIRED__condX1 <= 0
 c1611: - 15.8453086887446 sv_function_helpervar_b4,r7__Z3
        + sv_function_helpervar_b4,r7__condX3 <= 0
 c1612: 0.158453086908613 sv_function_helpervar_b3,r12__Z2
        - sv_function_helpervar_b3,r12__condX2 <= 0
 c1613: 461.611759739108 sv_function_helpervar_b8,r9__Z3
        + 249.888706346973 sv_function_helpervar_b8,r9__Z2 - aux_sv_b8,r9
        + 1636.3575441296 sv_function_helpervar_b8,r9__condX1
        + 59.3058675647 sv_function_helpervar_b8,r9__condX2
        + 31.4734419831 sv_function_helpervar_b8,r9__condX3  = 0
 c1614: 48.8576557725385 sv_function_helpervar_b1,r13__Z3
        + 22.8677438316296 sv_function_helpervar_b1,r13__Z2 - aux_sv_b1,r13
        + 2.5072375042 sv_function_helpervar_b1,r13__condX3
        + 6.6289041352 sv_function_helpervar_b1,r13__condX2
        + 150.9476060917 sv_function_helpervar_b1,r13__condX1  = 0
 c1615: - aux_cap_helper__b8,r11,band_LOW_PAIRED__condX1 <= 0
 c1616: - aux_cap_helper__b0,r0,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b0,r0,band_HIGH_PAIRED__condX1 <= 0
 c1617: aux_cap_helper__b7,r8,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b7,r8,band_HIGH_PAIRED__Z1 <= 0
 c1618: aux_cap_helper__b4,r0,band_UNPAIRED__condX1
        - aux_cap_helper__b4,r0,band_UNPAIRED__Z1 <= 0
 c1619: - 0.158453086943132 sv_function_helpervar_b1,r3__Z1
        + sv_function_helpervar_b1,r3__condX1 <= 0
 c1620: 0.158453086892038 sv_function_helpervar_b6,r5__Z2
        - sv_function_helpervar_b6,r5__condX2 <= 0
 c1621: - 2 aux_cap_helper__b5,r4,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b5,r4,band_HIGH_PAIRED__condX2 <= 0
 c1622: aux_cap_helper__b0,r6,band_UNPAIRED__condX1
        - aux_cap_helper__b0,r6,band_UNPAIRED__Z1 <= 0
 c1623: aux_cap_b5,r6,band_LOW_PAIRED - aux_c_b5,r6
        + aux_cap_b5,r6,band_HIGH_PAIRED + aux_cap_b5,r6,band_UNPAIRED  = 0
 c1624: aux_cap_helper__b2,r1,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b2,r1,band_HIGH_PAIRED__Z3 <= 0
 c1625: 14 GlobalBidder_hat{W}__i8,k1 + GlobalBidder_W_i_8 <= 27
 c1626: sv_function_helpervar_b8,r1__Z3 + sv_function_helpervar_b8,r1__Z2
        + sv_function_helpervar_b8,r1__Z1  = 1
 c1627: sv_function_helpervar_b8,r12__condX2
        + sv_function_helpervar_b8,r12__condX3
        + sv_function_helpervar_b8,r12__condX1 - aux_c_b8,r12  = 0
 c1628: aux_cap_helper__b7,r10,band_LOW_PAIRED__condX1
        - aux_cap_helper__b7,r10,band_LOW_PAIRED__Z1 <= 0
 c1629: - aux_sv_b1,r3 + 22.7702361005495 sv_function_helpervar_b1,r3__Z2
        + 50.9194369807968 sv_function_helpervar_b1,r3__Z3
        + 150.9476061115 sv_function_helpervar_b1,r3__condX1
        + 7.2442770074 sv_function_helpervar_b1,r3__condX2
        + 2.3771181579 sv_function_helpervar_b1,r3__condX3  = 0
 c1630: 0.888446608 aux_cap_helper__b2,r0,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b2,r0,band_UNPAIRED__condX1
        - aux_cap_b2,r0,band_UNPAIRED
        - 0.253841888 aux_cap_helper__b2,r0,band_UNPAIRED__Z2  = 0
 c1631: - X__b5,r13,band_LOW_PAIRED
        + aux_cap_helper__b5,r13,band_LOW_PAIRED__condX2
        + aux_cap_helper__b5,r13,band_LOW_PAIRED__condX1  = 0
 c1632: sv_function_helpervar_b6,r10__Z1 + sv_function_helpervar_b6,r10__Z3
        + sv_function_helpervar_b6,r10__Z2  = 1
 c1633: 0.888446608 aux_cap_helper__b8,r4,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b8,r4,band_UNPAIRED__condX1
        - aux_cap_b8,r4,band_UNPAIRED
        - 0.253841888 aux_cap_helper__b8,r4,band_UNPAIRED__Z2  = 0
 c1634: 0.0102040816326531 X__b9,r8,band_HIGH_PAIRED - GlobalBidder_W_ir__b9,r8
        + 0.0102040816326531 X__b9,r8,band_UNPAIRED
        + 0.0102040816326531 X__b9,r8,band_LOW_PAIRED <= 0
 c1635: aux_cap_helper__b2,r0,band_LOW_PAIRED__Z2
        + aux_cap_helper__b2,r0,band_LOW_PAIRED__Z1  = 1
 c1636: aux_cap_helper__b4,r11,band_UNPAIRED__Z2
        - aux_cap_helper__b4,r11,band_UNPAIRED__condX2 <= 0
 c1637: aux_cap_helper__b4,r12,band_LOW_PAIRED__Z2
        - aux_cap_helper__b4,r12,band_LOW_PAIRED__condX2 <= 0
 c1638: - X__b1,r0,band_LOW_PAIRED
        + aux_cap_helper__b1,r0,band_LOW_PAIRED__condX1
        + aux_cap_helper__b1,r0,band_LOW_PAIRED__condX2  = 0
 c1639: sv_function_helpervar_b9,r10__condX2
        - 6.87475654633498 sv_function_helpervar_b9,r10__Z2 <= 0
 c1640: aux_cap_helper__b6,r0,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b6,r0,band_HIGH_PAIRED__Z1 <= 0
 c1641: - aux_cap_helper__b1,r10,band_LOW_PAIRED__condX1 <= 0
 c1642: - aux_cap_helper__b7,r12,band_LOW_PAIRED__condX1 <= 0
 c1643: aux_cap_b1,r11,band_HIGH_PAIRED + aux_cap_b1,r11,band_UNPAIRED
        + aux_cap_b1,r11,band_LOW_PAIRED - aux_c_b1,r11  = 0
 c1644: - 3 aux_cap_helper__b2,r8,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b2,r8,band_HIGH_PAIRED__condX3 <= 0
 c1645: aux_cap_helper__b8,r8,band_LOW_PAIRED__Z2
        - aux_cap_helper__b8,r8,band_LOW_PAIRED__condX2 <= 0
 c1646: sv_function_helpervar_b4,r2__Z3 + sv_function_helpervar_b4,r2__Z1
        + sv_function_helpervar_b4,r2__Z2  = 1
 c1647: aux_cap_helper__b4,r5,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b4,r5,band_LOW_PAIRED__Z2 <= 0
 c1648: aux_cap_b8,r13,band_UNPAIRED + aux_cap_b8,r13,band_LOW_PAIRED
        - aux_c_b8,r13 + aux_cap_b8,r13,band_HIGH_PAIRED  = 0
 c1649: - aux_Omega_b4,r2 + 192.1267981861 aux_sv_b4,r2  = 0
 c1650: - aux_Omega_b6,r2 + 173.345150179367 aux_sv_b6,r2  = 0
 c1651: aux_cap_helper__b6,r6,band_UNPAIRED__Z1
        + aux_cap_helper__b6,r6,band_UNPAIRED__Z2  = 1
 c1652: - sv_function_helpervar_b7,r5__condX2
        + 0.158453086897541 sv_function_helpervar_b7,r5__Z2 <= 0
 c1653: aux_cap_helper__b9,r2,band_LOW_PAIRED__Z2
        + aux_cap_helper__b9,r2,band_LOW_PAIRED__Z1  = 1
 c1654: - aux_cap_helper__b4,r11,band_UNPAIRED__condX1 <= 0
 c1655: - 0.158453087021984 sv_function_helpervar_b8,r1__Z1
        + sv_function_helpervar_b8,r1__condX1 <= 0
 c1656: aux_cap_helper__b8,r2,band_UNPAIRED__Z2
        - aux_cap_helper__b8,r2,band_UNPAIRED__condX2 <= 0
 c1657: aux_cap_helper__b0,r6,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b0,r6,band_HIGH_PAIRED__Z1 <= 0
 c1658: - aux_cap_helper__b7,r2,band_HIGH_PAIRED__condX1 <= 0
 c1659: - aux_cap_helper__b8,r8,band_LOW_PAIRED__condX1 <= 0
 c1660: aux_cap_b3,r3,band_UNPAIRED + aux_cap_b3,r3,band_LOW_PAIRED
        - aux_c_b3,r3 + aux_cap_b3,r3,band_HIGH_PAIRED  = 0
 c1661: - aux_cap_helper__b3,r13,band_UNPAIRED__condX2
        + aux_cap_helper__b3,r13,band_UNPAIRED__Z2 <= 0
 c1662: - X__b5,r10,band_UNPAIRED
        + aux_cap_helper__b5,r10,band_UNPAIRED__condX2
        + aux_cap_helper__b5,r10,band_UNPAIRED__condX1  = 0
 c1663: - aux_cap_helper__b8,r2,band_UNPAIRED__condX1 <= 0
 c1664: - sv_function_helpervar_b8,r5__condX1 <= 0
 c1665: sv_function_helpervar_b1,r9__condX1
        - 0.158453086895131 sv_function_helpervar_b1,r9__Z1 <= 0
 c1666: aux_cap_helper__b7,r2,band_LOW_PAIRED__Z1
        + aux_cap_helper__b7,r2,band_LOW_PAIRED__Z2  = 1
 c1667: - 2 aux_cap_helper__b1,r1,band_LOW_PAIRED__Z2
        + aux_cap_helper__b1,r1,band_LOW_PAIRED__condX2 <= 0
 c1668: 2 aux_cap_helper__b2,r11,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b2,r11,band_HIGH_PAIRED__condX3 <= 0
 c1669: - aux_cap_b6,r13,band_UNPAIRED
        - 0.253841888 aux_cap_helper__b6,r13,band_UNPAIRED__Z2
        + 0.63460472 aux_cap_helper__b6,r13,band_UNPAIRED__condX1
        + 0.888446608 aux_cap_helper__b6,r13,band_UNPAIRED__condX2  = 0
 c1670: - aux_cap_helper__b7,r12,band_LOW_PAIRED__condX2
        + aux_cap_helper__b7,r12,band_LOW_PAIRED__Z2 <= 0
 c1671: - aux_c_b7,r10 + sv_function_helpervar_b7,r10__condX3
        + sv_function_helpervar_b7,r10__condX1
        + sv_function_helpervar_b7,r10__condX2  = 0
 c1672: - aux_cap_helper__b9,r1,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b9,r1,band_HIGH_PAIRED__Z2 <= 0
 c1673: - 2 aux_cap_helper__b9,r4,band_UNPAIRED__Z2
        + aux_cap_helper__b9,r4,band_UNPAIRED__condX2 <= 0
 c1674: GlobalBidder_W_i_7 + 13 GlobalBidder_hat{W}__i7,k2 >= 13
 c1675: 27.609440159 sv_function_helpervar_b8,r2__condX3
        + 70.1669075922 sv_function_helpervar_b8,r2__condX2
        + 1636.3575436649 sv_function_helpervar_b8,r2__condX1 - aux_sv_b8,r2
        + 522.838061415893 sv_function_helpervar_b8,r2__Z3
        + 248.167741024542 sv_function_helpervar_b8,r2__Z2  = 0
 c1676: - aux_sv_b7,r5 + 25.6190797607 sv_function_helpervar_b7,r5__condX3
        + 42.7969503006 sv_function_helpervar_b7,r5__condX2
        + 1255.6694400953 sv_function_helpervar_b7,r5__condX1
        + 192.183390021076 sv_function_helpervar_b7,r5__Z2
        + 330.964065115393 sv_function_helpervar_b7,r5__Z3  = 0
 c1677: - aux_cap_helper__b9,r7,band_HIGH_PAIRED__condX1 <= 0
 c1678: - aux_cap_helper__b9,r11,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b9,r11,band_HIGH_PAIRED__Z2 <= 0
 c1679: GlobalBidder_W_i_9 - 14 GlobalBidder_hat{W}__i9,k0 >= 0
 c1680: - sv_function_helpervar_b9,r0__condX3
        + 6.29021169207404 sv_function_helpervar_b9,r0__Z3 <= 0
 c1681: aux_cap_helper__b2,r5,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b2,r5,band_HIGH_PAIRED__Z1 <= 0
 c1682: aux_cap_helper__b9,r2,band_UNPAIRED__Z1
        + aux_cap_helper__b9,r2,band_UNPAIRED__Z2  = 1
 c1683: - aux_cap_helper__b4,r8,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b4,r8,band_HIGH_PAIRED__Z2 <= 0
 c1684: sv_function_helpervar_b9,r4__condX1
        - 0.158453086901837 sv_function_helpervar_b9,r4__Z1 <= 0
 c1685: - aux_cap_helper__b4,r12,band_LOW_PAIRED__condX1 <= 0
 c1686: aux_cap_helper__b1,r4,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b1,r4,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b1,r4,band_HIGH_PAIRED__condX3
        - X__b1,r4,band_HIGH_PAIRED  = 0
 c1687: aux_cap_helper__b8,r1,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b8,r1,band_HIGH_PAIRED__Z3 <= 0
 c1688: - 2 aux_cap_helper__b8,r4,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b8,r4,band_HIGH_PAIRED__condX2 <= 0
 c1689: aux_cap_helper__b5,r7,band_UNPAIRED__Z1
        + aux_cap_helper__b5,r7,band_UNPAIRED__Z2  = 1
 c1690: aux_cap_helper__b8,r5,band_LOW_PAIRED__Z2
        - aux_cap_helper__b8,r5,band_LOW_PAIRED__condX2 <= 0
 c1691: - aux_cap_helper__b4,r8,band_HIGH_PAIRED__condX1 <= 0
 c1692: - sv_function_helpervar_b9,r13__condX2
        + 0.158453086965069 sv_function_helpervar_b9,r13__Z2 <= 0
 c1693: - aux_sv_b1,r2 + 150.9476061266 sv_function_helpervar_b1,r2__condX1
        + 2.3498924384 sv_function_helpervar_b1,r2__condX3
        + 7.3975874374 sv_function_helpervar_b1,r2__condX2
        + 51.3508369112701 sv_function_helpervar_b1,r2__Z3
        + 22.745943589772 sv_function_helpervar_b1,r2__Z2  = 0
 c1694: aux_cap_helper__b4,r0,band_LOW_PAIRED__Z1
        + aux_cap_helper__b4,r0,band_LOW_PAIRED__Z2  = 1
 c1695: aux_cap_helper__b8,r8,band_UNPAIRED__Z1
        + aux_cap_helper__b8,r8,band_UNPAIRED__Z2  = 1
 c1696: aux_cap_b9,r13,band_HIGH_PAIRED + aux_cap_b9,r13,band_LOW_PAIRED
        + aux_cap_b9,r13,band_UNPAIRED - aux_c_b9,r13  = 0
 c1697: - aux_cap_helper__b9,r7,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b9,r7,band_HIGH_PAIRED__Z2 <= 0
 c1698: aux_cap_helper__b4,r13,band_UNPAIRED__Z2
        + aux_cap_helper__b4,r13,band_UNPAIRED__Z1  = 1
 c1699: - aux_cap_b4,r5,band_UNPAIRED
        - 0.253841888 aux_cap_helper__b4,r5,band_UNPAIRED__Z2
        + 0.888446608 aux_cap_helper__b4,r5,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b4,r5,band_UNPAIRED__condX1  = 0
 c1700: 0.888446608 aux_cap_helper__b7,r7,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b7,r7,band_UNPAIRED__condX1
        - 0.253841888 aux_cap_helper__b7,r7,band_UNPAIRED__Z2
        - aux_cap_b7,r7,band_UNPAIRED  = 0
 c1701: - 0.253841888 aux_cap_helper__b0,r13,band_UNPAIRED__Z2
        - aux_cap_b0,r13,band_UNPAIRED
        + 0.888446608 aux_cap_helper__b0,r13,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b0,r13,band_UNPAIRED__condX1  = 0
 c1702: - 2 aux_cap_helper__b6,r1,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b6,r1,band_HIGH_PAIRED__condX2 <= 0
 c1703: - aux_cap_helper__b5,r5,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b5,r5,band_HIGH_PAIRED__condX1 <= 0
 c1704: sv_function_helpervar_b3,r9__condX3
        + sv_function_helpervar_b3,r9__condX1
        + sv_function_helpervar_b3,r9__condX2 - aux_c_b3,r9  = 0
 c1705: aux_cap_helper__b5,r13,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b5,r13,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b5,r13,band_HIGH_PAIRED__Z2  = 1
 c1706: - aux_cap_helper__b7,r8,band_LOW_PAIRED__condX1 <= 0
 c1707: - X__b2,r6,band_HIGH_PAIRED
        + aux_cap_helper__b2,r6,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b2,r6,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b2,r6,band_HIGH_PAIRED__condX1  = 0
 c1708: aux_cap_helper__b5,r11,band_UNPAIRED__Z2
        + aux_cap_helper__b5,r11,band_UNPAIRED__Z1  = 1
 c1709: aux_cap_helper__b0,r5,band_LOW_PAIRED__condX1
        + aux_cap_helper__b0,r5,band_LOW_PAIRED__condX2
        - X__b0,r5,band_LOW_PAIRED  = 0
 c1710: - aux_cap_helper__b1,r9,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b1,r9,band_HIGH_PAIRED__Z2 <= 0
 c1711: - 0.158453087005849 sv_function_helpervar_b7,r9__Z1
        + sv_function_helpervar_b7,r9__condX1 <= 0
 c1712: sv_function_helpervar_b0,r2__Z2 + sv_function_helpervar_b0,r2__Z1
        + sv_function_helpervar_b0,r2__Z3  = 1
 c1713: aux_cap_helper__b6,r9,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b6,r9,band_LOW_PAIRED__Z2 <= 0
 c1714: 6.6034700582062 sv_function_helpervar_b8,r5__Z3
        - sv_function_helpervar_b8,r5__condX3 <= 0
 c1715: - aux_cap_helper__b3,r0,band_LOW_PAIRED__condX1 <= 0
 c1716: - 3 aux_cap_helper__b5,r2,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b5,r2,band_HIGH_PAIRED__condX3 <= 0
 c1717: - aux_sv_b8,r12 + 72.2364279898 sv_function_helpervar_b8,r12__condX2
        + 27.0891705149 sv_function_helpervar_b8,r12__condX3
        + 1636.3575438141 sv_function_helpervar_b8,r12__condX1
        + 247.839819130109 sv_function_helpervar_b8,r12__Z2
        + 531.081894527703 sv_function_helpervar_b8,r12__Z3  = 0
 c1718: aux_cap_helper__b7,r11,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b7,r11,band_HIGH_PAIRED__Z3 <= 0
 c1719: - aux_cap_helper__b9,r1,band_HIGH_PAIRED__condX1 <= 0
 c1720: aux_cap_helper__b1,r9,band_LOW_PAIRED__condX1
        - aux_cap_helper__b1,r9,band_LOW_PAIRED__Z1 <= 0
 c1721: - 15.8453086887446 sv_function_helpervar_b5,r13__Z3
        + sv_function_helpervar_b5,r13__condX3 <= 0
 c1722: - aux_cap_helper__b6,r1,band_LOW_PAIRED__Z1
        + aux_cap_helper__b6,r1,band_LOW_PAIRED__condX1 <= 0
 c1723: sv_function_helpervar_b6,r9__condX3
        + sv_function_helpervar_b6,r9__condX2
        + sv_function_helpervar_b6,r9__condX1 - aux_c_b6,r9  = 0
 c1724: - aux_cap_helper__b6,r7,band_HIGH_PAIRED__condX1 <= 0
 c1725: - aux_sv_b5,r13 + 320.8168253946 sv_function_helpervar_b5,r13__Z3
        + 148.128647094361 sv_function_helpervar_b5,r13__Z2
        + 978.9018731056 sv_function_helpervar_b5,r13__condX1
        + 16.0088230738 sv_function_helpervar_b5,r13__condX3
        + 44.0595800474 sv_function_helpervar_b5,r13__condX2  = 0
 c1726: - aux_cap_helper__b1,r9,band_HIGH_PAIRED__condX1 <= 0
 c1727: aux_cap_helper__b9,r13,band_LOW_PAIRED__Z2
        - aux_cap_helper__b9,r13,band_LOW_PAIRED__condX2 <= 0
 c1728: - aux_cap_helper__b1,r3,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b1,r3,band_HIGH_PAIRED__Z3 <= 0
 c1729: aux_cap_helper__b2,r2,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b2,r2,band_HIGH_PAIRED__Z3 <= 0
 c1730: - sv_function_helpervar_b5,r12__condX1 <= 0
 c1731: sv_function_helpervar_b5,r1__condX3
        - 15.8453086887446 sv_function_helpervar_b5,r1__Z3 <= 0
 c1732: - 2 aux_cap_helper__b7,r11,band_UNPAIRED__Z2
        + aux_cap_helper__b7,r11,band_UNPAIRED__condX2 <= 0
 c1733: - aux_cap_helper__b0,r13,band_UNPAIRED__Z1
        + aux_cap_helper__b0,r13,band_UNPAIRED__condX1 <= 0
 c1734: 0.158453086994015 sv_function_helpervar_b2,r9__Z2
        - sv_function_helpervar_b2,r9__condX2 <= 0
 c1735: aux_cap_helper__b6,r13,band_LOW_PAIRED__Z2
        - aux_cap_helper__b6,r13,band_LOW_PAIRED__condX2 <= 0
 c1736: - aux_cap_helper__b9,r13,band_LOW_PAIRED__condX1 <= 0
 c1737: - sv_function_helpervar_b7,r12__condX3
        + 6.80514243953476 sv_function_helpervar_b7,r12__Z3 <= 0
 c1738: - aux_Omega_b6,r12 + 743.597443521504 aux_sv_b6,r12  = 0
 c1739: - aux_cap_b9,r2,band_LOW_PAIRED
        - 1.2715511236 aux_cap_helper__b9,r2,band_LOW_PAIRED__Z2
        + 4.4504289328 aux_cap_helper__b9,r2,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b9,r2,band_LOW_PAIRED__condX1  = 0
 c1740: aux_cap_helper__b6,r6,band_UNPAIRED__condX1
        - aux_cap_helper__b6,r6,band_UNPAIRED__Z1 <= 0
 c1741: - aux_Omega_b7,r10 + 1172.96134128683 aux_sv_b7,r10  = 0
 c1742: aux_cap_helper__b9,r13,band_UNPAIRED__Z1
        + aux_cap_helper__b9,r13,band_UNPAIRED__Z2  = 1
 c1743: - aux_c_b8,r0 + sv_function_helpervar_b8,r0__condX3
        + sv_function_helpervar_b8,r0__condX1
        + sv_function_helpervar_b8,r0__condX2  = 0
 c1744: - 2 aux_cap_helper__b3,r5,band_LOW_PAIRED__Z2
        + aux_cap_helper__b3,r5,band_LOW_PAIRED__condX2 <= 0
 c1745: - aux_cap_helper__b6,r13,band_LOW_PAIRED__condX1 <= 0
 c1746: aux_cap_helper__b5,r4,band_LOW_PAIRED__Z2
        - aux_cap_helper__b5,r4,band_LOW_PAIRED__condX2 <= 0
 c1747: 0.0102040816326531 X__b7,r11,band_UNPAIRED
        + 0.0102040816326531 X__b7,r11,band_LOW_PAIRED
        + 0.0102040816326531 X__b7,r11,band_HIGH_PAIRED
        - GlobalBidder_W_ir__b7,r11 <= 0
 c1748: aux_cap_helper__b3,r1,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b3,r1,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b3,r1,band_HIGH_PAIRED__condX1
        - X__b3,r1,band_HIGH_PAIRED  = 0
 c1749: - aux_cap_helper__b8,r5,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b8,r5,band_HIGH_PAIRED__condX1 <= 0
 c1750: - aux_Omega_b4,r5 + 450.673056840812 aux_sv_b4,r5  = 0
 c1751: - aux_cap_helper__b8,r5,band_UNPAIRED__Z1
        + aux_cap_helper__b8,r5,band_UNPAIRED__condX1 <= 0
 c1752: aux_cap_helper__b3,r10,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b3,r10,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b3,r10,band_HIGH_PAIRED__Z3  = 1
 c1753: - aux_cap_helper__b3,r0,band_LOW_PAIRED__condX2
        + aux_cap_helper__b3,r0,band_LOW_PAIRED__Z2 <= 0
 c1754: - aux_cap_helper__b1,r9,band_UNPAIRED__condX1 <= 0
 c1755: aux_cap_helper__b8,r7,band_LOW_PAIRED__Z2
        + aux_cap_helper__b8,r7,band_LOW_PAIRED__Z1  = 1
 c1756: aux_cap_helper__b2,r3,band_UNPAIRED__condX2
        + aux_cap_helper__b2,r3,band_UNPAIRED__condX1 - X__b2,r3,band_UNPAIRED
         = 0
 c1757: - aux_cap_helper__b3,r8,band_UNPAIRED__condX2
        + aux_cap_helper__b3,r8,band_UNPAIRED__Z2 <= 0
 c1758: sv_function_helpervar_b3,r8__Z1 + sv_function_helpervar_b3,r8__Z3
        + sv_function_helpervar_b3,r8__Z2  = 1
 c1759: - aux_cap_helper__b3,r13,band_LOW_PAIRED__condX1 <= 0
 c1760: - 0.253841888 aux_cap_helper__b4,r7,band_UNPAIRED__Z2
        + 0.63460472 aux_cap_helper__b4,r7,band_UNPAIRED__condX1
        + 0.888446608 aux_cap_helper__b4,r7,band_UNPAIRED__condX2
        - aux_cap_b4,r7,band_UNPAIRED  = 0
 c1761: aux_cap_helper__b5,r5,band_UNPAIRED__condX1
        + aux_cap_helper__b5,r5,band_UNPAIRED__condX2 - X__b5,r5,band_UNPAIRED
         = 0
 c1762: aux_cap_helper__b4,r9,band_LOW_PAIRED__Z1
        + aux_cap_helper__b4,r9,band_LOW_PAIRED__Z2  = 1
 c1763: aux_cap_helper__b3,r1,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b3,r1,band_HIGH_PAIRED__Z2 <= 0
 c1764: - sv_function_helpervar_b1,r12__condX3
        + 5.6159506028775 sv_function_helpervar_b1,r12__Z3 <= 0
 c1765: - 3 aux_cap_helper__b8,r12,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b8,r12,band_HIGH_PAIRED__condX3 <= 0
 c1766: - 15.8453086887446 sv_function_helpervar_b2,r5__Z3
        + sv_function_helpervar_b2,r5__condX3 <= 0
 c1767: aux_cap_helper__b7,r1,band_UNPAIRED__condX1
        + aux_cap_helper__b7,r1,band_UNPAIRED__condX2 - X__b7,r1,band_UNPAIRED
         = 0
 c1768: - sv_function_helpervar_b1,r6__condX2
        + 0.158453086995604 sv_function_helpervar_b1,r6__Z2 <= 0
 c1769: 0.63460472 aux_cap_helper__b2,r2,band_UNPAIRED__condX1
        + 0.888446608 aux_cap_helper__b2,r2,band_UNPAIRED__condX2
        - 0.253841888 aux_cap_helper__b2,r2,band_UNPAIRED__Z2
        - aux_cap_b2,r2,band_UNPAIRED  = 0
 c1770: - X__b9,r5,band_LOW_PAIRED
        + aux_cap_helper__b9,r5,band_LOW_PAIRED__condX2
        + aux_cap_helper__b9,r5,band_LOW_PAIRED__condX1  = 0
 c1771: - sv_function_helpervar_b5,r0__condX1 <= 0
 c1772: - aux_cap_helper__b3,r13,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b3,r13,band_HIGH_PAIRED__Z3 <= 0
 c1773: sv_function_helpervar_b3,r7__condX2
        + sv_function_helpervar_b3,r7__condX1
        + sv_function_helpervar_b3,r7__condX3 - aux_c_b3,r7  = 0
 c1774: 3.1788778092 aux_cap_helper__b3,r1,band_LOW_PAIRED__condX1
        + 4.4504289328 aux_cap_helper__b3,r1,band_LOW_PAIRED__condX2
        - aux_cap_b3,r1,band_LOW_PAIRED
        - 1.2715511236 aux_cap_helper__b3,r1,band_LOW_PAIRED__Z2  = 0
 c1775: 0.0102040816326531 X__b7,r1,band_HIGH_PAIRED
        + 0.0102040816326531 X__b7,r1,band_LOW_PAIRED
        - GlobalBidder_W_ir__b7,r1 + 0.0102040816326531 X__b7,r1,band_UNPAIRED
        <= 0
 c1776: - aux_cap_helper__b0,r3,band_UNPAIRED__condX1 <= 0
 c1777: aux_cap_helper__b2,r5,band_UNPAIRED__Z2
        - aux_cap_helper__b2,r5,band_UNPAIRED__condX2 <= 0
 c1778: - aux_c_b4,r10 + sv_function_helpervar_b4,r10__condX1
        + sv_function_helpervar_b4,r10__condX3
        + sv_function_helpervar_b4,r10__condX2  = 0
 c1779: 20668026.8748422 GlobalBidder_hat{W}__i8,k1 - GlobalBidder_PSI__i8,k1
        + aux_Omega_b8,r12 + aux_Omega_b8,r13 + aux_Omega_b8,r10
        + aux_Omega_b8,r11 + aux_Omega_b8,r5 + aux_Omega_b8,r6
        + aux_Omega_b8,r7 + aux_Omega_b8,r8 + aux_Omega_b8,r9 + aux_Omega_b8,r0
        + aux_Omega_b8,r1 + aux_Omega_b8,r2 + aux_Omega_b8,r3 + aux_Omega_b8,r4
        <= 20668026.8748422
 c1780: 49.1617117274 sv_function_helpervar_b5,r1__condX2
        + 978.9018728422 sv_function_helpervar_b5,r1__condX1
        + 15.0424258669 sv_function_helpervar_b5,r1__condX3 - aux_sv_b5,r1
        + 336.129687453434 sv_function_helpervar_b5,r1__Z3
        + 147.320198577818 sv_function_helpervar_b5,r1__Z2  = 0
 c1781: aux_cap_helper__b1,r2,band_UNPAIRED__Z1
        + aux_cap_helper__b1,r2,band_UNPAIRED__Z2  = 1
 c1782: sv_function_helpervar_b4,r12__condX2
        + sv_function_helpervar_b4,r12__condX3
        + sv_function_helpervar_b4,r12__condX1 - aux_c_b4,r12  = 0
 c1783: - aux_cap_helper__b0,r4,band_LOW_PAIRED__Z1
        + aux_cap_helper__b0,r4,band_LOW_PAIRED__condX1 <= 0
 c1784: 26.4622103160252 aux_sv_b5,r13 - aux_Omega_b5,r13  = 0
 c1785: - aux_Omega_b3,r10 + 410.898977243479 aux_sv_b3,r10  = 0
 c1786: aux_cap_helper__b1,r9,band_UNPAIRED__Z2
        - aux_cap_helper__b1,r9,band_UNPAIRED__condX2 <= 0
 c1787: 1994.46488215272 aux_sv_b2,r11 - aux_Omega_b2,r11  = 0
 c1788: aux_cap_helper__b3,r12,band_LOW_PAIRED__Z1
        + aux_cap_helper__b3,r12,band_LOW_PAIRED__Z2  = 1
 c1789: - aux_cap_helper__b4,r4,band_UNPAIRED__condX2
        + aux_cap_helper__b4,r4,band_UNPAIRED__Z2 <= 0
 c1790: - aux_Omega_b6,r5 + 480.343836551081 aux_sv_b6,r5  = 0
 c1791: aux_cap_helper__b6,r12,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b6,r12,band_HIGH_PAIRED__Z2 <= 0
 c1792: 2 aux_cap_helper__b1,r10,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b1,r10,band_HIGH_PAIRED__condX3 <= 0
 c1793: 0.888446608 aux_cap_helper__b6,r11,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b6,r11,band_UNPAIRED__condX1
        - 0.253841888 aux_cap_helper__b6,r11,band_UNPAIRED__Z2
        - aux_cap_b6,r11,band_UNPAIRED  = 0
 c1794: - aux_cap_helper__b8,r5,band_LOW_PAIRED__condX1 <= 0
 c1795: - sv_function_helpervar_b6,r6__condX2
        + 0.158453086913347 sv_function_helpervar_b6,r6__Z2 <= 0
 c1796: - sv_function_helpervar_b6,r3__condX1 <= 0
 c1797: sv_function_helpervar_b8,r8__condX3
        + sv_function_helpervar_b8,r8__condX2
        + sv_function_helpervar_b8,r8__condX1 - aux_c_b8,r8  = 0
 c1798: 0.0102040816326531 X__b9,r4,band_UNPAIRED
        + 0.0102040816326531 X__b9,r4,band_HIGH_PAIRED
        - GlobalBidder_W_ir__b9,r4
        + 0.0102040816326531 X__b9,r4,band_LOW_PAIRED <= 0
 c1799: - aux_cap_b5,r2,band_UNPAIRED
        + 0.63460472 aux_cap_helper__b5,r2,band_UNPAIRED__condX1
        + 0.888446608 aux_cap_helper__b5,r2,band_UNPAIRED__condX2
        - 0.253841888 aux_cap_helper__b5,r2,band_UNPAIRED__Z2  = 0
 c1800: - aux_cap_helper__b5,r4,band_LOW_PAIRED__condX1 <= 0
 c1801: - aux_cap_b9,r10,band_HIGH_PAIRED
        - 0.8112667416 aux_cap_helper__b9,r10,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b9,r10,band_HIGH_PAIRED__Z3
        + 2.0281668543 aux_cap_helper__b9,r10,band_HIGH_PAIRED__condX1
        + 2.8394335959 aux_cap_helper__b9,r10,band_HIGH_PAIRED__condX2
        + 1.8253501688 aux_cap_helper__b9,r10,band_HIGH_PAIRED__condX3  = 0
 c1802: 893.54269237861 aux_sv_b5,r12 - aux_Omega_b5,r12  = 0
 c1803: - X__b9,r10,band_HIGH_PAIRED
        + aux_cap_helper__b9,r10,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b9,r10,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b9,r10,band_HIGH_PAIRED__condX3  = 0
 c1804: aux_cap_helper__b3,r8,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b3,r8,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b3,r8,band_HIGH_PAIRED__condX2
        - X__b3,r8,band_HIGH_PAIRED  = 0
 c1805: - aux_cap_helper__b5,r0,band_UNPAIRED__Z1
        + aux_cap_helper__b5,r0,band_UNPAIRED__condX1 <= 0
 c1806: aux_cap_helper__b7,r3,band_LOW_PAIRED__condX1
        - aux_cap_helper__b7,r3,band_LOW_PAIRED__Z1 <= 0
 c1807: - aux_cap_b4,r11,band_UNPAIRED
        - 0.253841888 aux_cap_helper__b4,r11,band_UNPAIRED__Z2
        + 0.888446608 aux_cap_helper__b4,r11,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b4,r11,band_UNPAIRED__condX1  = 0
 c1808: - aux_cap_helper__b3,r8,band_UNPAIRED__condX1 <= 0
 c1809: aux_cap_helper__b9,r1,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b9,r1,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b9,r1,band_HIGH_PAIRED__condX3
        - X__b9,r1,band_HIGH_PAIRED  = 0
 c1810: - aux_cap_helper__b9,r11,band_HIGH_PAIRED__condX1 <= 0
 c1811: sv_function_helpervar_b4,r9__condX2
        - 5.95597226282117 sv_function_helpervar_b4,r9__Z2 <= 0
 c1812: aux_cap_helper__b1,r6,band_LOW_PAIRED__Z2
        + aux_cap_helper__b1,r6,band_LOW_PAIRED__Z1  = 1
 c1813: X__b3,r10,band_UNPAIRED + X__b9,r10,band_UNPAIRED
        + X__b1,r10,band_UNPAIRED + X__b5,r10,band_UNPAIRED
        + X__b7,r10,band_UNPAIRED + X__b2,r10,band_UNPAIRED
        + X__b0,r10,band_UNPAIRED + X__b6,r10,band_UNPAIRED
        + X__b8,r10,band_UNPAIRED + X__b4,r10,band_UNPAIRED <= 2
 c1814: - 1.2715511236 aux_cap_helper__b5,r9,band_LOW_PAIRED__Z2
        + 4.4504289328 aux_cap_helper__b5,r9,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b5,r9,band_LOW_PAIRED__condX1
        - aux_cap_b5,r9,band_LOW_PAIRED  = 0
 c1815: aux_cap_helper__b8,r4,band_UNPAIRED__condX2
        + aux_cap_helper__b8,r4,band_UNPAIRED__condX1 - X__b8,r4,band_UNPAIRED
         = 0
 c1816: - 2 aux_cap_helper__b4,r2,band_LOW_PAIRED__Z2
        + aux_cap_helper__b4,r2,band_LOW_PAIRED__condX2 <= 0
 c1817: sv_function_helpervar_b3,r1__condX3
        + sv_function_helpervar_b3,r1__condX1
        + sv_function_helpervar_b3,r1__condX2 - aux_c_b3,r1  = 0
 c1818: - aux_cap_helper__b9,r1,band_UNPAIRED__Z1
        + aux_cap_helper__b9,r1,band_UNPAIRED__condX1 <= 0
 c1819: aux_cap_helper__b7,r13,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b7,r13,band_HIGH_PAIRED__Z2 <= 0
 c1820: aux_cap_helper__b9,r4,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b9,r4,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b9,r4,band_HIGH_PAIRED__Z3  = 1
 c1821: sv_function_helpervar_b1,r12__condX3
        - 15.8453086887446 sv_function_helpervar_b1,r12__Z3 <= 0
 c1822: 50.7427346556588 sv_function_helpervar_b1,r10__Z3
        + 22.7797441583817 sv_function_helpervar_b1,r10__Z2 - aux_sv_b1,r10
        + 150.9476061201 sv_function_helpervar_b1,r10__condX1
        + 7.1842715007 sv_function_helpervar_b1,r10__condX2
        + 2.3882698703 sv_function_helpervar_b1,r10__condX3  = 0
 c1823: - 2 aux_cap_helper__b5,r0,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b5,r0,band_HIGH_PAIRED__condX2 <= 0
 c1824: aux_cap_helper__b7,r6,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b7,r6,band_HIGH_PAIRED__Z2 <= 0
 c1825: aux_cap_helper__b8,r2,band_LOW_PAIRED__Z2
        - aux_cap_helper__b8,r2,band_LOW_PAIRED__condX2 <= 0
 c1826: 339.483747113208 sv_function_helpervar_b7,r10__Z3
        + 192.037788577747 sv_function_helpervar_b7,r10__Z2 - aux_sv_b7,r10
        + 25.0814012487 sv_function_helpervar_b7,r10__condX3
        + 1255.6694400362 sv_function_helpervar_b7,r10__condX1
        + 43.7158433676 sv_function_helpervar_b7,r10__condX2  = 0
 c1827: sv_function_helpervar_b2,r1__condX1
        - 0.15845308691457 sv_function_helpervar_b2,r1__Z1 <= 0
 c1828: sv_function_helpervar_b2,r0__condX2
        - 5.90029274601902 sv_function_helpervar_b2,r0__Z2 <= 0
 c1829: - sv_function_helpervar_b5,r11__condX1 <= 0
 c1830: sv_function_helpervar_b5,r4__condX2
        - 5.55364789890483 sv_function_helpervar_b5,r4__Z2 <= 0
 c1831: - X__b7,r4,band_UNPAIRED + aux_cap_helper__b7,r4,band_UNPAIRED__condX1
        + aux_cap_helper__b7,r4,band_UNPAIRED__condX2  = 0
 c1832: - X__b2,r9,band_HIGH_PAIRED
        + aux_cap_helper__b2,r9,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b2,r9,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b2,r9,band_HIGH_PAIRED__condX1  = 0
 c1833: 0.158453086934315 sv_function_helpervar_b2,r11__Z2
        - sv_function_helpervar_b2,r11__condX2 <= 0
 c1834: aux_cap_helper__b4,r12,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b4,r12,band_HIGH_PAIRED__condX2 <= 0
 c1835: 332.610580270072 sv_function_helpervar_b5,r11__Z3
        + 147.531711189101 sv_function_helpervar_b5,r11__Z2 - aux_sv_b5,r11
        + 978.9018728896 sv_function_helpervar_b5,r11__condX1
        + 15.2645172917 sv_function_helpervar_b5,r11__condX3
        + 47.8268522116 sv_function_helpervar_b5,r11__condX2  = 0
 c1836: - 3 aux_cap_helper__b1,r7,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b1,r7,band_HIGH_PAIRED__condX3 <= 0
 c1837: aux_cap_helper__b9,r9,band_LOW_PAIRED__Z2
        + aux_cap_helper__b9,r9,band_LOW_PAIRED__Z1  = 1
 c1838: - aux_cap_helper__b8,r9,band_UNPAIRED__condX1 <= 0
 c1839: aux_cap_helper__b0,r4,band_LOW_PAIRED__Z2
        + aux_cap_helper__b0,r4,band_LOW_PAIRED__Z1  = 1
 c1840: aux_cap_helper__b9,r1,band_LOW_PAIRED__Z2
        - aux_cap_helper__b9,r1,band_LOW_PAIRED__condX2 <= 0
 c1841: sv_function_helpervar_b9,r11__condX1
        - 0.158453086979881 sv_function_helpervar_b9,r11__Z1 <= 0
 c1842: 305.364717144311 sv_function_helpervar_b9,r2__Z2
        + 527.118833930242 sv_function_helpervar_b9,r2__Z3
        + 68.1308471208 sv_function_helpervar_b9,r2__condX2
        + 40.6331681732 sv_function_helpervar_b9,r2__condX3
        + 1995.2925256718 sv_function_helpervar_b9,r2__condX1 - aux_sv_b9,r2
         = 0
 c1843: - 1.2715511236 aux_cap_helper__b9,r1,band_LOW_PAIRED__Z2
        - aux_cap_b9,r1,band_LOW_PAIRED
        + 4.4504289328 aux_cap_helper__b9,r1,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b9,r1,band_LOW_PAIRED__condX1  = 0
 c1844: aux_cap_helper__b0,r10,band_LOW_PAIRED__Z1
        + aux_cap_helper__b0,r10,band_LOW_PAIRED__Z2  = 1
 c1845: - 1.2715511236 aux_cap_helper__b6,r11,band_LOW_PAIRED__Z2
        + 4.4504289328 aux_cap_helper__b6,r11,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b6,r11,band_LOW_PAIRED__condX1
        - aux_cap_b6,r11,band_LOW_PAIRED  = 0
 c1846: - aux_cap_helper__b9,r1,band_LOW_PAIRED__condX1 <= 0
 c1847: sv_function_helpervar_b3,r5__Z3 + sv_function_helpervar_b3,r5__Z2
        + sv_function_helpervar_b3,r5__Z1  = 1
 c1848: - aux_cap_b4,r6,band_LOW_PAIRED
        - 1.2715511236 aux_cap_helper__b4,r6,band_LOW_PAIRED__Z2
        + 4.4504289328 aux_cap_helper__b4,r6,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b4,r6,band_LOW_PAIRED__condX1  = 0
 c1849: 287.393567466434 sv_function_helpervar_b4,r7__Z3
        + 133.769857046237 sv_function_helpervar_b4,r7__Z2
        + 39.1790714455 sv_function_helpervar_b4,r7__condX2
        + 14.5811678794 sv_function_helpervar_b4,r7__condX3
        + 883.4028074615 sv_function_helpervar_b4,r7__condX1 - aux_sv_b4,r7
         = 0
 c1850: aux_cap_helper__b0,r13,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b0,r13,band_HIGH_PAIRED__Z3 <= 0
 c1851: aux_cap_helper__b7,r4,band_UNPAIRED__condX1
        - aux_cap_helper__b7,r4,band_UNPAIRED__Z1 <= 0
 c1852: - aux_c_b7,r5 + sv_function_helpervar_b7,r5__condX3
        + sv_function_helpervar_b7,r5__condX2
        + sv_function_helpervar_b7,r5__condX1  = 0
 c1853: 3.1788778092 aux_cap_helper__b2,r2,band_LOW_PAIRED__condX1
        + 4.4504289328 aux_cap_helper__b2,r2,band_LOW_PAIRED__condX2
        - 1.2715511236 aux_cap_helper__b2,r2,band_LOW_PAIRED__Z2
        - aux_cap_b2,r2,band_LOW_PAIRED  = 0
 c1854: sv_function_helpervar_b4,r1__Z2 + sv_function_helpervar_b4,r1__Z3
        + sv_function_helpervar_b4,r1__Z1  = 1
 c1855: sv_function_helpervar_b4,r4__condX2
        + sv_function_helpervar_b4,r4__condX1
        + sv_function_helpervar_b4,r4__condX3 - aux_c_b4,r4  = 0
 c1856: - X__b7,r10,band_HIGH_PAIRED
        + aux_cap_helper__b7,r10,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b7,r10,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b7,r10,band_HIGH_PAIRED__condX2  = 0
 c1857: sv_function_helpervar_b1,r6__condX2
        - 6.07000388758382 sv_function_helpervar_b1,r6__Z2 <= 0
 c1858: - aux_cap_helper__b6,r4,band_LOW_PAIRED__Z1
        + aux_cap_helper__b6,r4,band_LOW_PAIRED__condX1 <= 0
 c1859: - 2 aux_cap_helper__b0,r3,band_UNPAIRED__Z2
        + aux_cap_helper__b0,r3,band_UNPAIRED__condX2 <= 0
 c1860: - aux_cap_helper__b2,r3,band_HIGH_PAIRED__condX1 <= 0
 c1861: - aux_cap_helper__b2,r5,band_UNPAIRED__Z1
        + aux_cap_helper__b2,r5,band_UNPAIRED__condX1 <= 0
 c1862: 0.158453086933477 sv_function_helpervar_b3,r4__Z2
        - sv_function_helpervar_b3,r4__condX2 <= 0
 c1863: aux_cap_helper__b4,r2,band_UNPAIRED__condX1
        - aux_cap_helper__b4,r2,band_UNPAIRED__Z1 <= 0
 c1864: aux_cap_helper__b8,r9,band_UNPAIRED__Z2
        - aux_cap_helper__b8,r9,band_UNPAIRED__condX2 <= 0
 c1865: - sv_function_helpervar_b6,r11__condX3
        + 5.65397694524761 sv_function_helpervar_b6,r11__Z3 <= 0
 c1866: aux_cap_helper__b1,r11,band_UNPAIRED__condX1
        - aux_cap_helper__b1,r11,band_UNPAIRED__Z1 <= 0
 c1867: sv_function_helpervar_b8,r11__condX1
        - 0.158453087021465 sv_function_helpervar_b8,r11__Z1 <= 0
 c1868: - 2 aux_cap_helper__b0,r12,band_UNPAIRED__Z2
        + aux_cap_helper__b0,r12,band_UNPAIRED__condX2 <= 0
 c1869: aux_cap_helper__b1,r0,band_UNPAIRED__condX1
        - aux_cap_helper__b1,r0,band_UNPAIRED__Z1 <= 0
 c1870: - aux_cap_helper__b3,r7,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b3,r7,band_HIGH_PAIRED__Z2 <= 0
 c1871: - aux_cap_helper__b5,r7,band_UNPAIRED__condX2
        + aux_cap_helper__b5,r7,band_UNPAIRED__Z2 <= 0
 c1872: - aux_cap_helper__b8,r2,band_LOW_PAIRED__condX1 <= 0
 c1873: 0.63460472 aux_cap_helper__b7,r12,band_UNPAIRED__condX1
        + 0.888446608 aux_cap_helper__b7,r12,band_UNPAIRED__condX2
        - aux_cap_b7,r12,band_UNPAIRED
        - 0.253841888 aux_cap_helper__b7,r12,band_UNPAIRED__Z2  = 0
 c1874: X__b5,r8,band_LOW_PAIRED + X__b1,r8,band_LOW_PAIRED
        + X__b3,r8,band_LOW_PAIRED + X__b0,r8,band_LOW_PAIRED
        + X__b2,r8,band_LOW_PAIRED + X__b4,r8,band_LOW_PAIRED
        + X__b8,r8,band_LOW_PAIRED + X__b9,r8,band_LOW_PAIRED
        + X__b6,r8,band_LOW_PAIRED + X__b7,r8,band_LOW_PAIRED <= 2
 c1875: aux_cap_helper__b6,r8,band_UNPAIRED__Z2
        + aux_cap_helper__b6,r8,band_UNPAIRED__Z1  = 1
 c1876: sv_function_helpervar_b3,r1__Z3 + sv_function_helpervar_b3,r1__Z2
        + sv_function_helpervar_b3,r1__Z1  = 1
 c1877: aux_cap_helper__b0,r0,band_LOW_PAIRED__Z2
        - aux_cap_helper__b0,r0,band_LOW_PAIRED__condX2 <= 0
 c1878: sv_function_helpervar_b8,r2__condX3
        + sv_function_helpervar_b8,r2__condX2
        + sv_function_helpervar_b8,r2__condX1 - aux_c_b8,r2  = 0
 c1879: - X__b5,r11,band_LOW_PAIRED
        + aux_cap_helper__b5,r11,band_LOW_PAIRED__condX2
        + aux_cap_helper__b5,r11,band_LOW_PAIRED__condX1  = 0
 c1880: - X__b5,r2,band_UNPAIRED + aux_cap_helper__b5,r2,band_UNPAIRED__condX1
        + aux_cap_helper__b5,r2,band_UNPAIRED__condX2  = 0
 c1881: - 3 aux_cap_helper__b7,r0,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b7,r0,band_HIGH_PAIRED__condX3 <= 0
 c1882: - aux_cap_helper__b3,r7,band_HIGH_PAIRED__condX1 <= 0
 c1883: - aux_cap_helper__b5,r7,band_UNPAIRED__condX1 <= 0
 c1884: 621.674015311577 aux_sv_b8,r9 - aux_Omega_b8,r9  = 0
 c1885: aux_cap_helper__b9,r6,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b9,r6,band_HIGH_PAIRED__Z3 <= 0
 c1886: - aux_cap_helper__b0,r0,band_LOW_PAIRED__condX1 <= 0
 c1887: 34.0826482276 sv_function_helpervar_b8,r10__condX3
        + 54.6763675812 sv_function_helpervar_b8,r10__condX2
        + 1636.357543897 sv_function_helpervar_b8,r10__condX1 - aux_sv_b8,r10
        + 250.622264908916 sv_function_helpervar_b8,r10__Z2
        + 420.268081363033 sv_function_helpervar_b8,r10__Z3  = 0
 c1888: - aux_cap_helper__b1,r6,band_LOW_PAIRED__Z1
        + aux_cap_helper__b1,r6,band_LOW_PAIRED__condX1 <= 0
 c1889: aux_cap_helper__b1,r6,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b1,r6,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b1,r6,band_HIGH_PAIRED__Z1  = 1
 c1890: 174.056378749812 sv_function_helpervar_b3,r12__Z2
        + 383.8606849213 sv_function_helpervar_b3,r12__Z3
        + 18.4481098264 sv_function_helpervar_b3,r12__condX3
        + 53.7151088309 sv_function_helpervar_b3,r12__condX2
        + 1152.1877365696 sv_function_helpervar_b3,r12__condX1 - aux_sv_b3,r12
         = 0
 c1891: - 2 aux_cap_helper__b0,r13,band_LOW_PAIRED__Z2
        + aux_cap_helper__b0,r13,band_LOW_PAIRED__condX2 <= 0
 c1892: aux_cap_helper__b2,r12,band_LOW_PAIRED__Z1
        + aux_cap_helper__b2,r12,band_LOW_PAIRED__Z2  = 1
 c1893: - aux_cap_helper__b1,r1,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b1,r1,band_HIGH_PAIRED__condX1 <= 0
 c1894: 0.0102040816326531 X__b8,r5,band_HIGH_PAIRED
        + 0.0102040816326531 X__b8,r5,band_LOW_PAIRED
        + 0.0102040816326531 X__b8,r5,band_UNPAIRED - GlobalBidder_W_ir__b8,r5
        <= 0
 c1895: sv_function_helpervar_b7,r3__condX3
        - 15.8453086887446 sv_function_helpervar_b7,r3__Z3 <= 0
 c1896: aux_cap_helper__b1,r8,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b1,r8,band_UNPAIRED__Z2 <= 0
 c1897: sv_function_helpervar_b1,r11__condX2
        - 6.15161959363656 sv_function_helpervar_b1,r11__Z2 <= 0
 c1898: GlobalBidder_W_i_7 - 14 GlobalBidder_hat{W}__i7,k1 >= -1
 c1899: 2 aux_cap_helper__b5,r1,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b5,r1,band_HIGH_PAIRED__condX3 <= 0
 c1900: 0.158453086897641 sv_function_helpervar_b0,r10__Z2
        - sv_function_helpervar_b0,r10__condX2 <= 0
 c1901: - aux_cap_b3,r2,band_UNPAIRED
        + 0.888446608 aux_cap_helper__b3,r2,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b3,r2,band_UNPAIRED__condX1
        - 0.253841888 aux_cap_helper__b3,r2,band_UNPAIRED__Z2  = 0
 c1902: aux_cap_helper__b2,r9,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b2,r9,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b2,r9,band_HIGH_PAIRED__Z3  = 1
 c1903: aux_cap_helper__b4,r8,band_LOW_PAIRED__condX2
        + aux_cap_helper__b4,r8,band_LOW_PAIRED__condX1
        - X__b4,r8,band_LOW_PAIRED  = 0
 c1904: sv_function_helpervar_b0,r3__condX3
        - 15.8453086887446 sv_function_helpervar_b0,r3__Z3 <= 0
 c1905: - aux_cap_helper__b7,r5,band_HIGH_PAIRED__condX1 <= 0
 c1906: 0.63460472 aux_cap_helper__b9,r6,band_UNPAIRED__condX1
        + 0.888446608 aux_cap_helper__b9,r6,band_UNPAIRED__condX2
        - 0.253841888 aux_cap_helper__b9,r6,band_UNPAIRED__Z2
        - aux_cap_b9,r6,band_UNPAIRED  = 0
 c1907: - aux_cap_helper__b9,r0,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b9,r0,band_HIGH_PAIRED__condX1 <= 0
 c1908: - aux_c_b7,r12 + aux_cap_b7,r12,band_UNPAIRED
        + aux_cap_b7,r12,band_HIGH_PAIRED + aux_cap_b7,r12,band_LOW_PAIRED  = 0
 c1909: - aux_cap_b1,r5,band_UNPAIRED
        - 0.253841888 aux_cap_helper__b1,r5,band_UNPAIRED__Z2
        + 0.888446608 aux_cap_helper__b1,r5,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b1,r5,band_UNPAIRED__condX1  = 0
 c1910: aux_cap_b1,r2,band_HIGH_PAIRED + aux_cap_b1,r2,band_UNPAIRED
        - aux_c_b1,r2 + aux_cap_b1,r2,band_LOW_PAIRED  = 0
 c1911: - 2 aux_cap_helper__b1,r12,band_LOW_PAIRED__Z2
        + aux_cap_helper__b1,r12,band_LOW_PAIRED__condX2 <= 0
 c1912: - sv_function_helpervar_b8,r6__condX1 <= 0
 c1913: 2 aux_cap_helper__b3,r6,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b3,r6,band_HIGH_PAIRED__condX3 <= 0
 c1914: sv_function_helpervar_b2,r3__condX2
        - 6.30553906524748 sv_function_helpervar_b2,r3__Z2 <= 0
 c1915: aux_cap_helper__b9,r12,band_UNPAIRED__Z2
        - aux_cap_helper__b9,r12,band_UNPAIRED__condX2 <= 0
 c1916: - aux_cap_helper__b9,r10,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b9,r10,band_HIGH_PAIRED__condX1 <= 0
 c1917: - 15.8453086887446 sv_function_helpervar_b6,r8__Z3
        + sv_function_helpervar_b6,r8__condX3 <= 0
 c1918: sv_function_helpervar_b4,r5__Z1 + sv_function_helpervar_b4,r5__Z2
        + sv_function_helpervar_b4,r5__Z3  = 1
 c1919: aux_cap_helper__b7,r9,band_UNPAIRED__Z1
        + aux_cap_helper__b7,r9,band_UNPAIRED__Z2  = 1
 c1920: aux_cap_helper__b4,r11,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b4,r11,band_HIGH_PAIRED__Z1 <= 0
 c1921: - GlobalBidder_W_ir__b9,r13 + X__b9,r13,band_LOW_PAIRED
        + X__b9,r13,band_UNPAIRED + X__b9,r13,band_HIGH_PAIRED >= 0
 c1922: - aux_cap_helper__b4,r10,band_LOW_PAIRED__Z1
        + aux_cap_helper__b4,r10,band_LOW_PAIRED__condX1 <= 0
 c1923: - sv_function_helpervar_b2,r1__condX1 <= 0
 c1924: aux_cap_helper__b3,r4,band_LOW_PAIRED__condX1
        - aux_cap_helper__b3,r4,band_LOW_PAIRED__Z1 <= 0
 c1925: - aux_cap_helper__b5,r2,band_UNPAIRED__condX1 <= 0
 c1926: - 2 aux_cap_helper__b9,r9,band_UNPAIRED__Z2
        + aux_cap_helper__b9,r9,band_UNPAIRED__condX2 <= 0
 c1927: - aux_cap_helper__b9,r12,band_UNPAIRED__condX1 <= 0
 c1928: - aux_c_b0,r6 + sv_function_helpervar_b0,r6__condX3
        + sv_function_helpervar_b0,r6__condX2
        + sv_function_helpervar_b0,r6__condX1  = 0
 c1929: aux_cap_helper__b8,r9,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b8,r9,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b8,r9,band_HIGH_PAIRED__Z2  = 1
 c1930: aux_cap_b9,r3,band_HIGH_PAIRED + aux_cap_b9,r3,band_UNPAIRED
        + aux_cap_b9,r3,band_LOW_PAIRED - aux_c_b9,r3  = 0
 c1931: aux_cap_b7,r7,band_LOW_PAIRED - aux_c_b7,r7
        + aux_cap_b7,r7,band_UNPAIRED + aux_cap_b7,r7,band_HIGH_PAIRED  = 0
 c1932: - 2 aux_cap_helper__b2,r10,band_UNPAIRED__Z2
        + aux_cap_helper__b2,r10,band_UNPAIRED__condX2 <= 0
 c1933: aux_cap_helper__b7,r12,band_UNPAIRED__condX1
        + aux_cap_helper__b7,r12,band_UNPAIRED__condX2
        - X__b7,r12,band_UNPAIRED  = 0
 c1934: aux_cap_helper__b4,r9,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b4,r9,band_HIGH_PAIRED__Z2 <= 0
 c1935: aux_cap_helper__b4,r3,band_LOW_PAIRED__condX1
        - aux_cap_helper__b4,r3,band_LOW_PAIRED__Z1 <= 0
 c1936: - X__b5,r5,band_HIGH_PAIRED
        + aux_cap_helper__b5,r5,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b5,r5,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b5,r5,band_HIGH_PAIRED__condX2  = 0
 c1937: - aux_cap_helper__b2,r8,band_LOW_PAIRED__Z1
        + aux_cap_helper__b2,r8,band_LOW_PAIRED__condX1 <= 0
 c1938: aux_cap_helper__b7,r5,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b7,r5,band_HIGH_PAIRED__condX2 <= 0
 c1939: aux_cap_helper__b0,r3,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b0,r3,band_HIGH_PAIRED__Z3 <= 0
 c1940: aux_cap_helper__b3,r5,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b3,r5,band_UNPAIRED__Z2 <= 0
 c1941: sv_function_helpervar_b8,r11__condX3
        + sv_function_helpervar_b8,r11__condX1
        + sv_function_helpervar_b8,r11__condX2 - aux_c_b8,r11  = 0
 c1942: - sv_function_helpervar_b5,r9__condX3
        + 6.04014953766023 sv_function_helpervar_b5,r9__Z3 <= 0
 c1943: aux_cap_helper__b5,r2,band_LOW_PAIRED__condX2
        + aux_cap_helper__b5,r2,band_LOW_PAIRED__condX1
        - X__b5,r2,band_LOW_PAIRED  = 0
 c1944: sv_function_helpervar_b5,r10__condX2
        - 5.95863360031299 sv_function_helpervar_b5,r10__Z2 <= 0
 c1945: aux_cap_helper__b9,r11,band_LOW_PAIRED__Z2
        + aux_cap_helper__b9,r11,band_LOW_PAIRED__Z1  = 1
 c1946: aux_cap_helper__b5,r2,band_LOW_PAIRED__Z2
        + aux_cap_helper__b5,r2,band_LOW_PAIRED__Z1  = 1
 c1947: - 0.8112667416 aux_cap_helper__b8,r12,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b8,r12,band_HIGH_PAIRED__Z3
        - aux_cap_b8,r12,band_HIGH_PAIRED
        + 1.8253501688 aux_cap_helper__b8,r12,band_HIGH_PAIRED__condX3
        + 2.0281668543 aux_cap_helper__b8,r12,band_HIGH_PAIRED__condX1
        + 2.8394335959 aux_cap_helper__b8,r12,band_HIGH_PAIRED__condX2  = 0
 c1948: aux_cap_helper__b5,r12,band_LOW_PAIRED__condX2
        + aux_cap_helper__b5,r12,band_LOW_PAIRED__condX1
        - X__b5,r12,band_LOW_PAIRED  = 0
 c1949: - aux_cap_b4,r3,band_LOW_PAIRED
        + 3.1788778092 aux_cap_helper__b4,r3,band_LOW_PAIRED__condX1
        + 4.4504289328 aux_cap_helper__b4,r3,band_LOW_PAIRED__condX2
        - 1.2715511236 aux_cap_helper__b4,r3,band_LOW_PAIRED__Z2  = 0
 c1950: 2 aux_cap_helper__b8,r8,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b8,r8,band_HIGH_PAIRED__condX3 <= 0
 c1951: aux_cap_helper__b0,r8,band_UNPAIRED__condX2
        + aux_cap_helper__b0,r8,band_UNPAIRED__condX1 - X__b0,r8,band_UNPAIRED
         = 0
 c1952: aux_cap_helper__b2,r4,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b2,r4,band_HIGH_PAIRED__Z2 <= 0
 c1953: - aux_cap_helper__b5,r2,band_UNPAIRED__condX2
        + aux_cap_helper__b5,r2,band_UNPAIRED__Z2 <= 0
 c1954: - 15.8453086887446 sv_function_helpervar_b8,r1__Z3
        + sv_function_helpervar_b8,r1__condX3 <= 0
 c1955: - X__b8,r5,band_HIGH_PAIRED
        + aux_cap_helper__b8,r5,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b8,r5,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b8,r5,band_HIGH_PAIRED__condX3  = 0
 c1956: sv_function_helpervar_b9,r13__condX1
        - 0.158453086965069 sv_function_helpervar_b9,r13__Z1 <= 0
 c1957: X__b8,r2,band_UNPAIRED + X__b8,r2,band_LOW_PAIRED
        - GlobalBidder_W_ir__b8,r2 + X__b8,r2,band_HIGH_PAIRED >= 0
 c1958: aux_cap_b2,r6,band_HIGH_PAIRED - aux_c_b2,r6
        + aux_cap_b2,r6,band_UNPAIRED + aux_cap_b2,r6,band_LOW_PAIRED  = 0
 c1959: - aux_cap_helper__b1,r2,band_LOW_PAIRED__condX2
        + aux_cap_helper__b1,r2,band_LOW_PAIRED__Z2 <= 0
 c1960: X__b8,r3,band_HIGH_PAIRED + X__b8,r3,band_UNPAIRED
        + X__b8,r3,band_LOW_PAIRED - GlobalBidder_W_ir__b8,r3 >= 0
 c1961: aux_cap_helper__b2,r3,band_UNPAIRED__Z1
        + aux_cap_helper__b2,r3,band_UNPAIRED__Z2  = 1
 c1962: - aux_cap_helper__b7,r0,band_LOW_PAIRED__condX1 <= 0
 c1963: aux_cap_helper__b9,r1,band_UNPAIRED__Z1
        + aux_cap_helper__b9,r1,band_UNPAIRED__Z2  = 1
 c1964: 48.0168209318389 sv_function_helpervar_b1,r8__Z3
        + 22.9004752316029 sv_function_helpervar_b1,r8__Z2
        + 6.4223357395 sv_function_helpervar_b1,r8__condX2
        + 2.5603027273 sv_function_helpervar_b1,r8__condX3
        + 150.9476060917 sv_function_helpervar_b1,r8__condX1 - aux_sv_b1,r8
         = 0
 c1965: - aux_cap_helper__b5,r5,band_LOW_PAIRED__Z1
        + aux_cap_helper__b5,r5,band_LOW_PAIRED__condX1 <= 0
 c1966: aux_cap_helper__b1,r11,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b1,r11,band_HIGH_PAIRED__Z1 <= 0
 c1967: - aux_cap_helper__b6,r3,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b6,r3,band_HIGH_PAIRED__condX1 <= 0
 c1968: 8.06446672130127 sv_function_helpervar_b9,r2__Z3
        - sv_function_helpervar_b9,r2__condX3 <= 0
 c1969: - aux_cap_helper__b1,r2,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b1,r2,band_HIGH_PAIRED__Z2 <= 0
 c1970: aux_cap_helper__b6,r4,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b6,r4,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b6,r4,band_HIGH_PAIRED__condX1
        - X__b6,r4,band_HIGH_PAIRED  = 0
 c1971: - aux_cap_helper__b3,r11,band_LOW_PAIRED__Z1
        + aux_cap_helper__b3,r11,band_LOW_PAIRED__condX1 <= 0
 c1972: - sv_function_helpervar_b4,r2__condX3
        + 6.17498435770839 sv_function_helpervar_b4,r2__Z3 <= 0
 c1973: sv_function_helpervar_b3,r13__condX3
        - 15.8453086887446 sv_function_helpervar_b3,r13__Z3 <= 0
 c1974: - 5.91005559387013 sv_function_helpervar_b0,r11__Z2
        + sv_function_helpervar_b0,r11__condX2 <= 0
 c1975: aux_cap_helper__b6,r9,band_UNPAIRED__condX1
        + aux_cap_helper__b6,r9,band_UNPAIRED__condX2 - X__b6,r9,band_UNPAIRED
         = 0
 c1976: - aux_cap_helper__b1,r2,band_HIGH_PAIRED__condX1 <= 0
 c1977: aux_cap_helper__b9,r4,band_LOW_PAIRED__Z2
        - aux_cap_helper__b9,r4,band_LOW_PAIRED__condX2 <= 0
 c1978: aux_cap_helper__b1,r11,band_LOW_PAIRED__condX2
        + aux_cap_helper__b1,r11,band_LOW_PAIRED__condX1
        - X__b1,r11,band_LOW_PAIRED  = 0
 c1979: X__b1,r3,band_LOW_PAIRED + X__b7,r3,band_LOW_PAIRED
        + X__b8,r3,band_LOW_PAIRED + X__b9,r3,band_LOW_PAIRED
        + X__b0,r3,band_LOW_PAIRED + X__b6,r3,band_LOW_PAIRED
        + X__b5,r3,band_LOW_PAIRED + X__b4,r3,band_LOW_PAIRED
        + X__b3,r3,band_LOW_PAIRED + X__b2,r3,band_LOW_PAIRED <= 2
 c1980: - sv_function_helpervar_b7,r12__condX1 <= 0
 c1981: aux_cap_helper__b3,r5,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b3,r5,band_HIGH_PAIRED__Z2 <= 0
 c1982: sv_function_helpervar_b0,r12__condX1
        - 0.158453086902941 sv_function_helpervar_b0,r12__Z1 <= 0
 c1983: - 2 aux_cap_helper__b0,r10,band_LOW_PAIRED__Z2
        + aux_cap_helper__b0,r10,band_LOW_PAIRED__condX2 <= 0
 c1984: aux_cap_helper__b7,r0,band_LOW_PAIRED__Z2
        - aux_cap_helper__b7,r0,band_LOW_PAIRED__condX2 <= 0
 c1985: sv_function_helpervar_b4,r9__Z1 + sv_function_helpervar_b4,r9__Z3
        + sv_function_helpervar_b4,r9__Z2  = 1
 c1986: aux_cap_helper__b5,r12,band_LOW_PAIRED__condX1
        - aux_cap_helper__b5,r12,band_LOW_PAIRED__Z1 <= 0
 c1987: - 2 aux_cap_helper__b6,r2,band_UNPAIRED__Z2
        + aux_cap_helper__b6,r2,band_UNPAIRED__condX2 <= 0
 c1988: aux_cap_helper__b8,r13,band_UNPAIRED__Z2
        + aux_cap_helper__b8,r13,band_UNPAIRED__Z1  = 1
 c1989: - aux_sv_b0,r2 + 2.6020406162 sv_function_helpervar_b0,r2__condX3
        + 163.7425743899 sv_function_helpervar_b0,r2__condX1
        + 7.733779112 sv_function_helpervar_b0,r2__condX2
        + 24.7200751959913 sv_function_helpervar_b0,r2__Z2
        + 54.8643682891957 sv_function_helpervar_b0,r2__Z3  = 0
 c1990: - sv_function_helpervar_b6,r7__condX3
        + 6.04011244526223 sv_function_helpervar_b6,r7__Z3 <= 0
 c1991: - X__b0,r2,band_HIGH_PAIRED
        + aux_cap_helper__b0,r2,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b0,r2,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b0,r2,band_HIGH_PAIRED__condX1  = 0
 c1992: aux_cap_b2,r5,band_HIGH_PAIRED + aux_cap_b2,r5,band_UNPAIRED
        + aux_cap_b2,r5,band_LOW_PAIRED - aux_c_b2,r5  = 0
 c1993: aux_cap_helper__b4,r8,band_UNPAIRED__condX1
        + aux_cap_helper__b4,r8,band_UNPAIRED__condX2 - X__b4,r8,band_UNPAIRED
         = 0
 c1994: - 0.158453086953718 sv_function_helpervar_b5,r7__Z1
        + sv_function_helpervar_b5,r7__condX1 <= 0
 c1995: - sv_function_helpervar_b7,r2__condX3
        + 8.2076340013385 sv_function_helpervar_b7,r2__Z3 <= 0
 c1996: - 2 aux_cap_helper__b2,r11,band_LOW_PAIRED__Z2
        + aux_cap_helper__b2,r11,band_LOW_PAIRED__condX2 <= 0
 c1997: sv_function_helpervar_b4,r12__Z1 + sv_function_helpervar_b4,r12__Z2
        + sv_function_helpervar_b4,r12__Z3  = 1
 c1998: 647.728596781868 sv_function_helpervar_b9,r12__Z3
        + 302.197048366681 sv_function_helpervar_b9,r12__Z2 - aux_sv_b9,r12
        + 88.122056168 sv_function_helpervar_b9,r12__condX2
        + 1995.2925269695 sv_function_helpervar_b9,r12__condX1
        + 33.0214664878 sv_function_helpervar_b9,r12__condX3  = 0
 c1999: 0.00552 aux_Omega_b5,r1 + 0.01314 aux_Omega_b5,r0
        + 0.17678 aux_Omega_b5,r9 + 0.07433 aux_Omega_b5,r8
        + 0.03125 aux_Omega_b5,r7 + 0.03125 aux_Omega_b5,r6
        + 0.03125 aux_Omega_b5,r5 + 0.01314 aux_Omega_b5,r4
        + 0.01314 aux_Omega_b5,r3 + 0.00552 aux_Omega_b5,r2 - v__5
        + 0.42045 aux_Omega_b5,r13 + 0.42045 aux_Omega_b5,r10
        + aux_Omega_b5,r11 + 0.42045 aux_Omega_b5,r12  = 0
 c2000: aux_cap_helper__b8,r1,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b8,r1,band_UNPAIRED__Z2 <= 0
 c2001: sv_function_helpervar_b0,r10__condX1
        + sv_function_helpervar_b0,r10__condX3
        + sv_function_helpervar_b0,r10__condX2 - aux_c_b0,r10  = 0
 c2002: aux_cap_helper__b8,r0,band_UNPAIRED__Z1
        + aux_cap_helper__b8,r0,band_UNPAIRED__Z2  = 1
 c2003: aux_cap_helper__b4,r7,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b4,r7,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b4,r7,band_HIGH_PAIRED__Z3  = 1
 c2004: - 6.30568026667668 sv_function_helpervar_b1,r13__Z2
        + sv_function_helpervar_b1,r13__condX2 <= 0
 c2005: aux_cap_helper__b5,r12,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b5,r12,band_HIGH_PAIRED__Z3 <= 0
 c2006: aux_cap_helper__b6,r10,band_UNPAIRED__condX1
        - aux_cap_helper__b6,r10,band_UNPAIRED__Z1 <= 0
 c2007: X__b8,r13,band_UNPAIRED + X__b8,r13,band_LOW_PAIRED
        - GlobalBidder_W_ir__b8,r13 + X__b8,r13,band_HIGH_PAIRED >= 0
 c2008: 2.0281668543 aux_cap_helper__b0,r10,band_HIGH_PAIRED__condX1
        + 2.8394335959 aux_cap_helper__b0,r10,band_HIGH_PAIRED__condX2
        + 1.8253501688 aux_cap_helper__b0,r10,band_HIGH_PAIRED__condX3
        - aux_cap_b0,r10,band_HIGH_PAIRED
        - 0.8112667416 aux_cap_helper__b0,r10,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b0,r10,band_HIGH_PAIRED__Z3  = 0
 c2009: - 0.158453086921147 sv_function_helpervar_b2,r13__Z1
        + sv_function_helpervar_b2,r13__condX1 <= 0
 c2010: - aux_cap_helper__b7,r4,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b7,r4,band_HIGH_PAIRED__condX1 <= 0
 c2011: aux_cap_b2,r11,band_UNPAIRED + aux_cap_b2,r11,band_HIGH_PAIRED
        - aux_c_b2,r11 + aux_cap_b2,r11,band_LOW_PAIRED  = 0
 c2012: sv_function_helpervar_b0,r3__condX1
        - 0.158453086889895 sv_function_helpervar_b0,r3__Z1 <= 0
 c2013: - aux_cap_helper__b7,r6,band_LOW_PAIRED__Z1
        + aux_cap_helper__b7,r6,band_LOW_PAIRED__condX1 <= 0
 c2014: 0.0102040816326531 X__b9,r11,band_UNPAIRED - GlobalBidder_W_ir__b9,r11
        + 0.0102040816326531 X__b9,r11,band_HIGH_PAIRED
        + 0.0102040816326531 X__b9,r11,band_LOW_PAIRED <= 0
 c2015: - aux_cap_helper__b1,r12,band_HIGH_PAIRED__condX1 <= 0
 c2016: aux_cap_helper__b3,r10,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b3,r10,band_LOW_PAIRED__Z2 <= 0
 c2017: sv_function_helpervar_b9,r5__condX3
        - 15.8453086887446 sv_function_helpervar_b9,r5__Z3 <= 0
 c2018: - X__b1,r6,band_LOW_PAIRED
        + aux_cap_helper__b1,r6,band_LOW_PAIRED__condX1
        + aux_cap_helper__b1,r6,band_LOW_PAIRED__condX2  = 0
 c2019: 0.158453086903028 sv_function_helpervar_b9,r8__Z2
        - sv_function_helpervar_b9,r8__condX2 <= 0
 c2020: - 6.6326479692379 sv_function_helpervar_b1,r1__Z2
        + sv_function_helpervar_b1,r1__condX2 <= 0
 c2021: aux_cap_helper__b9,r0,band_LOW_PAIRED__Z1
        + aux_cap_helper__b9,r0,band_LOW_PAIRED__Z2  = 1
 c2022: - sv_function_helpervar_b1,r12__condX2
        + 0.158453086906349 sv_function_helpervar_b1,r12__Z2 <= 0
 c2023: aux_cap_helper__b3,r4,band_UNPAIRED__Z1
        + aux_cap_helper__b3,r4,band_UNPAIRED__Z2  = 1
 c2024: - aux_Omega_b4,r8 + 151.935009782789 aux_sv_b4,r8  = 0
 c2025: aux_cap_helper__b0,r9,band_UNPAIRED__Z2
        + aux_cap_helper__b0,r9,band_UNPAIRED__Z1  = 1
 c2026: - 1.2715511236 aux_cap_helper__b8,r13,band_LOW_PAIRED__Z2
        - aux_cap_b8,r13,band_LOW_PAIRED
        + 4.4504289328 aux_cap_helper__b8,r13,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b8,r13,band_LOW_PAIRED__condX1  = 0
 c2027: - aux_cap_helper__b9,r4,band_LOW_PAIRED__condX1 <= 0
 c2028: - 2 aux_cap_helper__b4,r13,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b4,r13,band_HIGH_PAIRED__condX2 <= 0
 c2029: 2 aux_cap_helper__b4,r10,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b4,r10,band_HIGH_PAIRED__condX3 <= 0
 c2030: sv_function_helpervar_b6,r1__condX2
        - 5.56357622273766 sv_function_helpervar_b6,r1__Z2 <= 0
 c2031: aux_cap_b0,r9,band_HIGH_PAIRED + aux_cap_b0,r9,band_UNPAIRED
        + aux_cap_b0,r9,band_LOW_PAIRED - aux_c_b0,r9  = 0
 c2032: - 1.2715511236 aux_cap_helper__b0,r10,band_LOW_PAIRED__Z2
        + 4.4504289328 aux_cap_helper__b0,r10,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b0,r10,band_LOW_PAIRED__condX1
        - aux_cap_b0,r10,band_LOW_PAIRED  = 0
 c2033: aux_cap_helper__b3,r13,band_UNPAIRED__Z2
        + aux_cap_helper__b3,r13,band_UNPAIRED__Z1  = 1
 c2034: aux_cap_helper__b3,r0,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b3,r0,band_UNPAIRED__Z2 <= 0
 c2035: aux_cap_helper__b5,r3,band_LOW_PAIRED__condX2
        + aux_cap_helper__b5,r3,band_LOW_PAIRED__condX1
        - X__b5,r3,band_LOW_PAIRED  = 0
 c2036: 2.8394335959 aux_cap_helper__b6,r13,band_HIGH_PAIRED__condX2
        + 1.8253501688 aux_cap_helper__b6,r13,band_HIGH_PAIRED__condX3
        + 2.0281668543 aux_cap_helper__b6,r13,band_HIGH_PAIRED__condX1
        - 0.8112667416 aux_cap_helper__b6,r13,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b6,r13,band_HIGH_PAIRED__Z3
        - aux_cap_b6,r13,band_HIGH_PAIRED  = 0
 c2037: - sv_function_helpervar_b4,r9__condX2
        + 0.158453086968442 sv_function_helpervar_b4,r9__Z2 <= 0
 c2038: aux_cap_helper__b1,r12,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b1,r12,band_HIGH_PAIRED__condX2 <= 0
 c2039: - aux_cap_helper__b6,r11,band_LOW_PAIRED__Z1
        + aux_cap_helper__b6,r11,band_LOW_PAIRED__condX1 <= 0
 c2040: aux_cap_b4,r2,band_UNPAIRED + aux_cap_b4,r2,band_HIGH_PAIRED
        + aux_cap_b4,r2,band_LOW_PAIRED - aux_c_b4,r2  = 0
 c2041: aux_cap_helper__b6,r13,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b6,r13,band_HIGH_PAIRED__Z3 <= 0
 c2042: - sv_function_helpervar_b5,r0__condX3
        + 6.0671903614491 sv_function_helpervar_b5,r0__Z3 <= 0
 c2043: - aux_cap_helper__b9,r12,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b9,r12,band_HIGH_PAIRED__Z3 <= 0
 c2044: - aux_cap_helper__b2,r13,band_HIGH_PAIRED__condX1 <= 0
 c2045: aux_cap_helper__b4,r13,band_LOW_PAIRED__condX1
        - aux_cap_helper__b4,r13,band_LOW_PAIRED__Z1 <= 0
 c2046: aux_cap_helper__b4,r9,band_LOW_PAIRED__condX1
        - aux_cap_helper__b4,r9,band_LOW_PAIRED__Z1 <= 0
 c2047: aux_cap_helper__b0,r10,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b0,r10,band_UNPAIRED__Z2 <= 0
 c2048: aux_cap_helper__b2,r4,band_LOW_PAIRED__Z2
        + aux_cap_helper__b2,r4,band_LOW_PAIRED__Z1  = 1
 c2049: aux_cap_helper__b8,r6,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b8,r6,band_UNPAIRED__Z2 <= 0
 c2050: - aux_cap_helper__b3,r7,band_LOW_PAIRED__Z1
        + aux_cap_helper__b3,r7,band_LOW_PAIRED__condX1 <= 0
 c2051: - 15.8453086887446 sv_function_helpervar_b4,r6__Z3
        + sv_function_helpervar_b4,r6__condX3 <= 0
 c2052: - X__b3,r10,band_UNPAIRED
        + aux_cap_helper__b3,r10,band_UNPAIRED__condX2
        + aux_cap_helper__b3,r10,band_UNPAIRED__condX1  = 0
 c2053: - aux_cap_helper__b7,r7,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b7,r7,band_HIGH_PAIRED__Z3 <= 0
 c2054: aux_cap_helper__b4,r5,band_UNPAIRED__Z1
        + aux_cap_helper__b4,r5,band_UNPAIRED__Z2  = 1
 c2055: aux_cap_helper__b6,r2,band_LOW_PAIRED__Z2
        + aux_cap_helper__b6,r2,band_LOW_PAIRED__Z1  = 1
 c2056: - aux_cap_helper__b7,r12,band_HIGH_PAIRED__condX1 <= 0
 c2057: - 0.158453087025632 sv_function_helpervar_b4,r6__Z1
        + sv_function_helpervar_b4,r6__condX1 <= 0
 c2058: - 0.253841888 aux_cap_helper__b6,r4,band_UNPAIRED__Z2
        + 0.888446608 aux_cap_helper__b6,r4,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b6,r4,band_UNPAIRED__condX1
        - aux_cap_b6,r4,band_UNPAIRED  = 0
 c2059: - aux_cap_helper__b2,r5,band_LOW_PAIRED__Z1
        + aux_cap_helper__b2,r5,band_LOW_PAIRED__condX1 <= 0
 c2060: - sv_function_helpervar_b4,r3__condX1 <= 0
 c2061: aux_cap_helper__b1,r3,band_LOW_PAIRED__condX1
        - aux_cap_helper__b1,r3,band_LOW_PAIRED__Z1 <= 0
 c2062: - aux_cap_helper__b8,r4,band_UNPAIRED__condX1 <= 0
 c2063: aux_cap_helper__b3,r7,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b3,r7,band_UNPAIRED__Z2 <= 0
 c2064: aux_cap_helper__b5,r4,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b5,r4,band_UNPAIRED__Z2 <= 0
 c2065: 47.4476660143914 sv_function_helpervar_b1,r1__Z3
        + 22.9207909172695 sv_function_helpervar_b1,r1__Z2
        + 6.2941231169 sv_function_helpervar_b1,r1__condX2
        + 150.9476060673 sv_function_helpervar_b1,r1__condX1
        + 2.5962221864 sv_function_helpervar_b1,r1__condX3 - aux_sv_b1,r1  = 0
 c2066: - aux_cap_helper__b2,r6,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b2,r6,band_HIGH_PAIRED__Z2 <= 0
 c2067: - sv_function_helpervar_b8,r4__condX2
        + 0.158453086919509 sv_function_helpervar_b8,r4__Z2 <= 0
 c2068: aux_cap_helper__b0,r10,band_UNPAIRED__Z2
        + aux_cap_helper__b0,r10,band_UNPAIRED__Z1  = 1
 c2069: - aux_c_b0,r11 + sv_function_helpervar_b0,r11__condX2
        + sv_function_helpervar_b0,r11__condX1
        + sv_function_helpervar_b0,r11__condX3  = 0
 c2070: - aux_cap_helper__b7,r12,band_UNPAIRED__condX2
        + aux_cap_helper__b7,r12,band_UNPAIRED__Z2 <= 0
 c2071: sv_function_helpervar_b9,r0__condX1
        - 0.158453086901946 sv_function_helpervar_b9,r0__Z1 <= 0
 c2072: 0.63460472 aux_cap_helper__b3,r5,band_UNPAIRED__condX1
        + 0.888446608 aux_cap_helper__b3,r5,band_UNPAIRED__condX2
        - 0.253841888 aux_cap_helper__b3,r5,band_UNPAIRED__Z2
        - aux_cap_b3,r5,band_UNPAIRED  = 0
 c2073: - aux_cap_helper__b2,r6,band_HIGH_PAIRED__condX1 <= 0
 c2074: 787.61529202214 aux_sv_b9,r10 - aux_Omega_b9,r10  = 0
 c2075: - sv_function_helpervar_b2,r9__condX1 <= 0
 c2076: aux_Omega_b7,r10 + aux_Omega_b7,r11 + aux_Omega_b7,r12
        + aux_Omega_b7,r13 + aux_Omega_b7,r0 + aux_Omega_b7,r1
        + aux_Omega_b7,r2 + aux_Omega_b7,r3 + aux_Omega_b7,r4 + aux_Omega_b7,r5
        + aux_Omega_b7,r6 + aux_Omega_b7,r7 + aux_Omega_b7,r8 + aux_Omega_b7,r9
        - GlobalBidder_PSI__i7,k0 + 20668026.8748422 GlobalBidder_hat{W}__i7,k0
        <= 20668026.8748422
 c2077: aux_cap_helper__b2,r13,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b2,r13,band_HIGH_PAIRED__condX2 <= 0
 c2078: aux_cap_helper__b2,r1,band_LOW_PAIRED__Z2
        - aux_cap_helper__b2,r1,band_LOW_PAIRED__condX2 <= 0
 c2079: - aux_cap_helper__b7,r12,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b7,r12,band_HIGH_PAIRED__Z2 <= 0
 c2080: - aux_cap_helper__b7,r12,band_UNPAIRED__condX1 <= 0
 c2081: 0.158453086905199 sv_function_helpervar_b8,r3__Z2
        - sv_function_helpervar_b8,r3__condX2 <= 0
 c2082: aux_cap_helper__b4,r11,band_UNPAIRED__Z2
        + aux_cap_helper__b4,r11,band_UNPAIRED__Z1  = 1
 c2083: - aux_cap_helper__b2,r3,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b2,r3,band_HIGH_PAIRED__Z2 <= 0
 c2084: - aux_cap_helper__b8,r4,band_UNPAIRED__condX2
        + aux_cap_helper__b8,r4,band_UNPAIRED__Z2 <= 0
 c2085: - aux_cap_helper__b2,r1,band_LOW_PAIRED__condX1 <= 0
 c2086: - 2 aux_cap_helper__b6,r5,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b6,r5,band_HIGH_PAIRED__condX2 <= 0
 c2087: aux_Omega_b2,r9 + aux_Omega_b2,r8 + aux_Omega_b2,r6 + aux_Omega_b2,r4
        + aux_Omega_b2,r0 - v__2  = 0
 c2088: - aux_cap_helper__b4,r0,band_LOW_PAIRED__Z1
        + aux_cap_helper__b4,r0,band_LOW_PAIRED__condX1 <= 0
 c2089: aux_cap_helper__b5,r2,band_LOW_PAIRED__condX1
        - aux_cap_helper__b5,r2,band_LOW_PAIRED__Z1 <= 0
 c2090: - 0.158453086887582 sv_function_helpervar_b5,r12__Z1
        + sv_function_helpervar_b5,r12__condX1 <= 0
 c2091: - aux_cap_helper__b6,r2,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b6,r2,band_HIGH_PAIRED__Z3 <= 0
 c2092: - 2 aux_cap_helper__b2,r9,band_LOW_PAIRED__Z2
        + aux_cap_helper__b2,r9,band_LOW_PAIRED__condX2 <= 0
 c2093: sv_function_helpervar_b2,r1__condX2
        - 6.0897224156964 sv_function_helpervar_b2,r1__Z2 <= 0
 c2094: - aux_cap_helper__b8,r6,band_HIGH_PAIRED__condX1 <= 0
 c2095: 2 aux_cap_helper__b4,r7,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b4,r7,band_HIGH_PAIRED__condX3 <= 0
 c2096: - sv_function_helpervar_b7,r0__condX1 <= 0
 c2097: - 5.78636564086377 sv_function_helpervar_b5,r2__Z2
        + sv_function_helpervar_b5,r2__condX2 <= 0
 c2098: sv_function_helpervar_b8,r11__condX3
        - 15.8453086887446 sv_function_helpervar_b8,r11__Z3 <= 0
 c2099: aux_cap_helper__b0,r10,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b0,r10,band_HIGH_PAIRED__Z3 <= 0
 c2100: - sv_function_helpervar_b0,r13__condX1 <= 0
 c2101: - aux_Omega_b1,r3 + 309.852933009955 aux_sv_b1,r3  = 0
 c2102: sv_function_helpervar_b9,r5__condX1
        + sv_function_helpervar_b9,r5__condX2
        + sv_function_helpervar_b9,r5__condX3 - aux_c_b9,r5  = 0
 c2103: 4.4504289328 aux_cap_helper__b6,r7,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b6,r7,band_LOW_PAIRED__condX1
        - 1.2715511236 aux_cap_helper__b6,r7,band_LOW_PAIRED__Z2
        - aux_cap_b6,r7,band_LOW_PAIRED  = 0
 c2104: - X__b1,r7,band_HIGH_PAIRED
        + aux_cap_helper__b1,r7,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b1,r7,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b1,r7,band_HIGH_PAIRED__condX1  = 0
 c2105: - aux_cap_helper__b2,r12,band_LOW_PAIRED__Z1
        + aux_cap_helper__b2,r12,band_LOW_PAIRED__condX1 <= 0
 c2106: - 2 aux_cap_helper__b3,r8,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b3,r8,band_HIGH_PAIRED__condX2 <= 0
 c2107: - 2 aux_cap_helper__b6,r7,band_UNPAIRED__Z2
        + aux_cap_helper__b6,r7,band_UNPAIRED__condX2 <= 0
 c2108: 0.0102040816326531 X__b7,r2,band_HIGH_PAIRED - GlobalBidder_W_ir__b7,r2
        + 0.0102040816326531 X__b7,r2,band_LOW_PAIRED
        + 0.0102040816326531 X__b7,r2,band_UNPAIRED <= 0
 c2109: sv_function_helpervar_b5,r10__condX1
        - 0.158453086987074 sv_function_helpervar_b5,r10__Z1 <= 0
 c2110: - 3 aux_cap_helper__b4,r4,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b4,r4,band_HIGH_PAIRED__condX3 <= 0
 c2111: aux_cap_helper__b7,r6,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b7,r6,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b7,r6,band_HIGH_PAIRED__Z3  = 1
 c2112: sv_function_helpervar_b3,r3__condX1
        - 0.158453086906449 sv_function_helpervar_b3,r3__Z1 <= 0
 c2113: 2 aux_cap_helper__b9,r2,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b9,r2,band_HIGH_PAIRED__condX3 <= 0
 c2114: aux_cap_helper__b0,r8,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b0,r8,band_HIGH_PAIRED__Z2 <= 0
 c2115: aux_cap_helper__b5,r11,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b5,r11,band_UNPAIRED__Z2 <= 0
 c2116: - 3 aux_cap_helper__b1,r4,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b1,r4,band_HIGH_PAIRED__condX3 <= 0
 c2117: aux_cap_helper__b8,r6,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b8,r6,band_HIGH_PAIRED__condX2 <= 0
 c2118: - aux_cap_helper__b8,r9,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b8,r9,band_HIGH_PAIRED__condX1 <= 0
 c2119: - aux_cap_helper__b0,r6,band_LOW_PAIRED__condX1 <= 0
 c2120: - aux_cap_helper__b5,r9,band_UNPAIRED__condX1 <= 0
 c2121: - aux_cap_helper__b5,r3,band_HIGH_PAIRED__condX1 <= 0
 c2122: - aux_cap_helper__b7,r10,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b7,r10,band_HIGH_PAIRED__Z3 <= 0
 c2123: aux_cap_helper__b0,r1,band_UNPAIRED__Z2
        - aux_cap_helper__b0,r1,band_UNPAIRED__condX2 <= 0
 c2124: aux_cap_helper__b3,r1,band_UNPAIRED__condX2
        + aux_cap_helper__b3,r1,band_UNPAIRED__condX1 - X__b3,r1,band_UNPAIRED
         = 0
 c2125: - aux_Omega_b1,r4 + 1212.11978933146 aux_sv_b1,r4  = 0
 c2126: - aux_Omega_b3,r4 + 1573.23190083209 aux_sv_b3,r4  = 0
 c2127: - aux_cap_helper__b9,r3,band_UNPAIRED__Z1
        + aux_cap_helper__b9,r3,band_UNPAIRED__condX1 <= 0
 c2128: - aux_cap_helper__b2,r7,band_UNPAIRED__condX1 <= 0
 c2129: aux_cap_helper__b8,r0,band_LOW_PAIRED__Z2
        + aux_cap_helper__b8,r0,band_LOW_PAIRED__Z1  = 1
 c2130: - aux_cap_helper__b9,r9,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b9,r9,band_HIGH_PAIRED__Z3 <= 0
 c2131: - aux_cap_helper__b1,r5,band_LOW_PAIRED__condX1 <= 0
 c2132: - sv_function_helpervar_b7,r13__condX3
        + 7.48392491398104 sv_function_helpervar_b7,r13__Z3 <= 0
 c2133: - X__b6,r5,band_LOW_PAIRED
        + aux_cap_helper__b6,r5,band_LOW_PAIRED__condX2
        + aux_cap_helper__b6,r5,band_LOW_PAIRED__condX1  = 0
 c2134: - aux_cap_b1,r12,band_UNPAIRED
        + 0.888446608 aux_cap_helper__b1,r12,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b1,r12,band_UNPAIRED__condX1
        - 0.253841888 aux_cap_helper__b1,r12,band_UNPAIRED__Z2  = 0
 c2135: 1.21690011250031 aux_cap_helper__b7,r2,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b7,r2,band_HIGH_PAIRED__Z2
        - aux_cap_b7,r2,band_HIGH_PAIRED
        + 1.8253501688 aux_cap_helper__b7,r2,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b7,r2,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b7,r2,band_HIGH_PAIRED__condX1  = 0
 c2136: aux_cap_helper__b9,r6,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b9,r6,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b9,r6,band_HIGH_PAIRED__Z2  = 1
 c2137: - aux_cap_helper__b6,r1,band_UNPAIRED__Z1
        + aux_cap_helper__b6,r1,band_UNPAIRED__condX1 <= 0
 c2138: - aux_cap_helper__b5,r13,band_HIGH_PAIRED__condX1 <= 0
 c2139: 0.0102040816326531 X__b8,r0,band_LOW_PAIRED
        + 0.0102040816326531 X__b8,r0,band_HIGH_PAIRED
        + 0.0102040816326531 X__b8,r0,band_UNPAIRED - GlobalBidder_W_ir__b8,r0
        <= 0
 c2140: aux_cap_helper__b8,r12,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b8,r12,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b8,r12,band_HIGH_PAIRED__Z3  = 1
 c2141: aux_cap_helper__b7,r3,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b7,r3,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b7,r3,band_HIGH_PAIRED__condX1
        - X__b7,r3,band_HIGH_PAIRED  = 0
 c2142: aux_cap_helper__b1,r7,band_LOW_PAIRED__condX1
        + aux_cap_helper__b1,r7,band_LOW_PAIRED__condX2
        - X__b1,r7,band_LOW_PAIRED  = 0
 c2143: aux_cap_helper__b6,r5,band_UNPAIRED__Z2
        - aux_cap_helper__b6,r5,band_UNPAIRED__condX2 <= 0
 c2144: sv_function_helpervar_b9,r1__Z3 + sv_function_helpervar_b9,r1__Z1
        + sv_function_helpervar_b9,r1__Z2  = 1
 c2145: aux_cap_helper__b2,r7,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b2,r7,band_HIGH_PAIRED__Z2 <= 0
 c2146: - aux_cap_b6,r10,band_HIGH_PAIRED
        + 1.8253501688 aux_cap_helper__b6,r10,band_HIGH_PAIRED__condX3
        + 2.0281668543 aux_cap_helper__b6,r10,band_HIGH_PAIRED__condX1
        + 2.8394335959 aux_cap_helper__b6,r10,band_HIGH_PAIRED__condX2
        + 1.21690011250031 aux_cap_helper__b6,r10,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b6,r10,band_HIGH_PAIRED__Z2  = 0
 c2147: aux_cap_helper__b0,r11,band_UNPAIRED__condX2
        + aux_cap_helper__b0,r11,band_UNPAIRED__condX1
        - X__b0,r11,band_UNPAIRED  = 0
 c2148: sv_function_helpervar_b2,r8__Z1 + sv_function_helpervar_b2,r8__Z2
        + sv_function_helpervar_b2,r8__Z3  = 1
 c2149: - aux_cap_helper__b0,r6,band_LOW_PAIRED__condX2
        + aux_cap_helper__b0,r6,band_LOW_PAIRED__Z2 <= 0
 c2150: aux_cap_helper__b6,r11,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b6,r11,band_UNPAIRED__Z2 <= 0
 c2151: - aux_cap_helper__b5,r9,band_UNPAIRED__condX2
        + aux_cap_helper__b5,r9,band_UNPAIRED__Z2 <= 0
 c2152: aux_cap_helper__b5,r3,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b5,r3,band_HIGH_PAIRED__condX2 <= 0
 c2153: - X__b9,r5,band_UNPAIRED + aux_cap_helper__b9,r5,band_UNPAIRED__condX2
        + aux_cap_helper__b9,r5,band_UNPAIRED__condX1  = 0
 c2154: - aux_Omega_b4,r4 + 1500.17507077137 aux_sv_b4,r4  = 0
 c2155: aux_cap_helper__b9,r12,band_UNPAIRED__Z2
        + aux_cap_helper__b9,r12,band_UNPAIRED__Z1  = 1
 c2156: aux_cap_b3,r9,band_UNPAIRED + aux_cap_b3,r9,band_LOW_PAIRED
        - aux_c_b3,r9 + aux_cap_b3,r9,band_HIGH_PAIRED  = 0
 c2157: aux_cap_helper__b2,r7,band_UNPAIRED__Z2
        - aux_cap_helper__b2,r7,band_UNPAIRED__condX2 <= 0
 c2158: aux_cap_helper__b3,r4,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b3,r4,band_HIGH_PAIRED__condX2 <= 0
 c2159: aux_cap_helper__b8,r11,band_LOW_PAIRED__Z2
        + aux_cap_helper__b8,r11,band_LOW_PAIRED__Z1  = 1
 c2160: aux_cap_b4,r7,band_HIGH_PAIRED - aux_c_b4,r7
        + aux_cap_b4,r7,band_LOW_PAIRED + aux_cap_b4,r7,band_UNPAIRED  = 0
 c2161: - sv_function_helpervar_b7,r2__condX2
        + 0.158453086902389 sv_function_helpervar_b7,r2__Z2 <= 0
 c2162: - aux_cap_helper__b1,r5,band_LOW_PAIRED__condX2
        + aux_cap_helper__b1,r5,band_LOW_PAIRED__Z2 <= 0
 c2163: - aux_cap_helper__b1,r13,band_UNPAIRED__condX2
        + aux_cap_helper__b1,r13,band_UNPAIRED__Z2 <= 0
 c2164: - aux_cap_helper__b3,r3,band_UNPAIRED__condX1 <= 0
 c2165: - aux_c_b7,r1 + aux_cap_b7,r1,band_HIGH_PAIRED
        + aux_cap_b7,r1,band_LOW_PAIRED + aux_cap_b7,r1,band_UNPAIRED  = 0
 c2166: aux_cap_helper__b5,r13,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b5,r13,band_HIGH_PAIRED__condX2 <= 0
 c2167: - aux_cap_helper__b1,r13,band_LOW_PAIRED__Z1
        + aux_cap_helper__b1,r13,band_LOW_PAIRED__condX1 <= 0
 c2168: - 0.8112667416 aux_cap_helper__b5,r2,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b5,r2,band_HIGH_PAIRED__Z3
        + 1.8253501688 aux_cap_helper__b5,r2,band_HIGH_PAIRED__condX3
        + 2.0281668543 aux_cap_helper__b5,r2,band_HIGH_PAIRED__condX1
        + 2.8394335959 aux_cap_helper__b5,r2,band_HIGH_PAIRED__condX2
        - aux_cap_b5,r2,band_HIGH_PAIRED  = 0
 c2169: aux_cap_helper__b4,r6,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b4,r6,band_HIGH_PAIRED__Z2 <= 0
 c2170: sv_function_helpervar_b4,r3__condX1
        + sv_function_helpervar_b4,r3__condX3
        + sv_function_helpervar_b4,r3__condX2 - aux_c_b4,r3  = 0
 c2171: aux_cap_helper__b3,r10,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b3,r10,band_HIGH_PAIRED__Z1 <= 0
 c2172: - aux_cap_helper__b0,r1,band_UNPAIRED__condX1 <= 0
 c2173: - 3 aux_cap_helper__b2,r12,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b2,r12,band_HIGH_PAIRED__condX3 <= 0
 c2174: - aux_cap_helper__b8,r7,band_UNPAIRED__Z1
        + aux_cap_helper__b8,r7,band_UNPAIRED__condX1 <= 0
 c2175: aux_cap_helper__b1,r8,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b1,r8,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b1,r8,band_HIGH_PAIRED__Z3  = 1
 c2176: aux_cap_helper__b9,r7,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b9,r7,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b9,r7,band_HIGH_PAIRED__condX3
        - X__b9,r7,band_HIGH_PAIRED  = 0
 c2177: - sv_function_helpervar_b6,r3__condX2
        + 0.158453086945764 sv_function_helpervar_b6,r3__Z2 <= 0
 c2178: aux_cap_helper__b0,r9,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b0,r9,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b0,r9,band_HIGH_PAIRED__Z3  = 1
 c2179: - X__b2,r13,band_HIGH_PAIRED
        + aux_cap_helper__b2,r13,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b2,r13,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b2,r13,band_HIGH_PAIRED__condX1  = 0
 c2180: - aux_cap_b7,r4,band_UNPAIRED
        + 0.63460472 aux_cap_helper__b7,r4,band_UNPAIRED__condX1
        + 0.888446608 aux_cap_helper__b7,r4,band_UNPAIRED__condX2
        - 0.253841888 aux_cap_helper__b7,r4,band_UNPAIRED__Z2  = 0
 c2181: 0.63460472 aux_cap_helper__b4,r2,band_UNPAIRED__condX1
        + 0.888446608 aux_cap_helper__b4,r2,band_UNPAIRED__condX2
        - aux_cap_b4,r2,band_UNPAIRED
        - 0.253841888 aux_cap_helper__b4,r2,band_UNPAIRED__Z2  = 0
 c2182: - aux_cap_helper__b7,r9,band_LOW_PAIRED__Z1
        + aux_cap_helper__b7,r9,band_LOW_PAIRED__condX1 <= 0
 c2183: sv_function_helpervar_b8,r1__condX2
        + sv_function_helpervar_b8,r1__condX3
        + sv_function_helpervar_b8,r1__condX1 - aux_c_b8,r1  = 0
 c2184: - X__b1,r10,band_LOW_PAIRED
        + aux_cap_helper__b1,r10,band_LOW_PAIRED__condX2
        + aux_cap_helper__b1,r10,band_LOW_PAIRED__condX1  = 0
 c2185: aux_cap_helper__b1,r1,band_UNPAIRED__Z1
        + aux_cap_helper__b1,r1,band_UNPAIRED__Z2  = 1
 c2186: - sv_function_helpervar_b8,r0__condX2
        + 0.158453086899447 sv_function_helpervar_b8,r0__Z2 <= 0
 c2187: aux_cap_helper__b8,r1,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b8,r1,band_LOW_PAIRED__Z2 <= 0
 c2188: aux_cap_helper__b3,r1,band_LOW_PAIRED__condX1
        - aux_cap_helper__b3,r1,band_LOW_PAIRED__Z1 <= 0
 c2189: - aux_Omega_b7,r12 + 1627.59651360757 aux_sv_b7,r12  = 0
 c2190: aux_Omega_b9,r1 + aux_Omega_b9,r0 + aux_Omega_b9,r3 + aux_Omega_b9,r2
        + aux_Omega_b9,r5 + aux_Omega_b9,r4 + aux_Omega_b9,r7 + aux_Omega_b9,r6
        + aux_Omega_b9,r9 + aux_Omega_b9,r8 - GlobalBidder_PSI__i9,k0
        + aux_Omega_b9,r13 + aux_Omega_b9,r10 + aux_Omega_b9,r11
        + aux_Omega_b9,r12 + 20668026.8748422 GlobalBidder_hat{W}__i9,k0
        <= 20668026.8748422
 c2191: 1.21690011250031 aux_cap_helper__b0,r1,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b0,r1,band_HIGH_PAIRED__Z2
        + 2.8394335959 aux_cap_helper__b0,r1,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b0,r1,band_HIGH_PAIRED__condX1
        + 1.8253501688 aux_cap_helper__b0,r1,band_HIGH_PAIRED__condX3
        - aux_cap_b0,r1,band_HIGH_PAIRED  = 0
 c2192: - aux_cap_helper__b3,r3,band_UNPAIRED__condX2
        + aux_cap_helper__b3,r3,band_UNPAIRED__Z2 <= 0
 c2193: aux_cap_helper__b6,r6,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b6,r6,band_HIGH_PAIRED__Z3 <= 0
 c2194: - aux_cap_helper__b4,r0,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b4,r0,band_HIGH_PAIRED__Z3 <= 0
 c2195: aux_cap_helper__b9,r9,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b9,r9,band_LOW_PAIRED__Z2 <= 0
 c2196: 4040.86213702191 aux_sv_b9,r11 - aux_Omega_b9,r11  = 0
 c2197: sv_function_helpervar_b7,r12__Z3 + sv_function_helpervar_b7,r12__Z1
        + sv_function_helpervar_b7,r12__Z2  = 1
 c2198: - 5.55639826435479 sv_function_helpervar_b0,r1__Z2
        + sv_function_helpervar_b0,r1__condX2 <= 0
 c2199: sv_function_helpervar_b1,r7__condX3
        - 15.8453086887446 sv_function_helpervar_b1,r7__Z3 <= 0
 c2200: - aux_Omega_b6,r8 + 176.192755885528 aux_sv_b6,r8  = 0
 c2201: 6.04618933446524 sv_function_helpervar_b0,r6__Z3
        - sv_function_helpervar_b0,r6__condX3 <= 0
 c2202: - aux_Omega_b2,r4 + 1519.02113348352 aux_sv_b2,r4  = 0
 c2203: - aux_cap_helper__b9,r10,band_UNPAIRED__condX1 <= 0
 c2204: - sv_function_helpervar_b9,r6__condX3
        + 7.9811679679664 sv_function_helpervar_b9,r6__Z3 <= 0
 c2205: aux_cap_helper__b2,r7,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b2,r7,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b2,r7,band_HIGH_PAIRED__Z3  = 1
 c2206: - X__b9,r13,band_LOW_PAIRED
        + aux_cap_helper__b9,r13,band_LOW_PAIRED__condX2
        + aux_cap_helper__b9,r13,band_LOW_PAIRED__condX1  = 0
 c2207: - aux_cap_helper__b1,r13,band_UNPAIRED__condX1 <= 0
 c2208: - X__b6,r6,band_UNPAIRED + aux_cap_helper__b6,r6,band_UNPAIRED__condX1
        + aux_cap_helper__b6,r6,band_UNPAIRED__condX2  = 0
 c2209: - aux_Omega_b7,r0 + 208.530341343575 aux_sv_b7,r0  = 0
 c2210: sv_function_helpervar_b5,r4__condX1
        - 0.158453086914415 sv_function_helpervar_b5,r4__Z1 <= 0
 c2211: aux_cap_helper__b6,r0,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b6,r0,band_LOW_PAIRED__Z2 <= 0
 c2212: aux_cap_helper__b7,r2,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b7,r2,band_LOW_PAIRED__Z2 <= 0
 c2213: - aux_cap_helper__b7,r1,band_UNPAIRED__condX2
        + aux_cap_helper__b7,r1,band_UNPAIRED__Z2 <= 0
 c2214: - aux_cap_b6,r7,band_UNPAIRED
        - 0.253841888 aux_cap_helper__b6,r7,band_UNPAIRED__Z2
        + 0.888446608 aux_cap_helper__b6,r7,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b6,r7,band_UNPAIRED__condX1  = 0
 c2215: 2 aux_cap_helper__b2,r8,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b2,r8,band_HIGH_PAIRED__condX3 <= 0
 c2216: 3.1788778092 aux_cap_helper__b7,r1,band_LOW_PAIRED__condX1
        + 4.4504289328 aux_cap_helper__b7,r1,band_LOW_PAIRED__condX2
        - 1.2715511236 aux_cap_helper__b7,r1,band_LOW_PAIRED__Z2
        - aux_cap_b7,r1,band_LOW_PAIRED  = 0
 c2217: - aux_cap_helper__b1,r0,band_LOW_PAIRED__Z1
        + aux_cap_helper__b1,r0,band_LOW_PAIRED__condX1 <= 0
 c2218: aux_cap_helper__b3,r2,band_LOW_PAIRED__Z1
        + aux_cap_helper__b3,r2,band_LOW_PAIRED__Z2  = 1
 c2219: - aux_cap_helper__b3,r11,band_HIGH_PAIRED__condX1 <= 0
 c2220: - X__b4,r7,band_LOW_PAIRED
        + aux_cap_helper__b4,r7,band_LOW_PAIRED__condX2
        + aux_cap_helper__b4,r7,band_LOW_PAIRED__condX1  = 0
 c2221: - 3 aux_cap_helper__b8,r5,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b8,r5,band_HIGH_PAIRED__condX3 <= 0
 c2222: - sv_function_helpervar_b5,r8__condX1 <= 0
 c2223: aux_cap_helper__b6,r7,band_LOW_PAIRED__condX1
        - aux_cap_helper__b6,r7,band_LOW_PAIRED__Z1 <= 0
 c2224: - aux_cap_helper__b9,r13,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b9,r13,band_HIGH_PAIRED__condX1 <= 0
 c2225: - sv_function_helpervar_b4,r2__condX1 <= 0
 c2226: - aux_cap_helper__b7,r1,band_UNPAIRED__condX1 <= 0
 c2227: GlobalBidder_W_i_9 + 14 GlobalBidder_hat{W}__i9,k2 <= 26
 c2228: sv_function_helpervar_b2,r4__condX3
        - 15.8453086887446 sv_function_helpervar_b2,r4__Z3 <= 0
 c2229: aux_cap_helper__b7,r9,band_LOW_PAIRED__Z2
        + aux_cap_helper__b7,r9,band_LOW_PAIRED__Z1  = 1
 c2230: - aux_cap_b9,r13,band_HIGH_PAIRED
        + 1.21690011250031 aux_cap_helper__b9,r13,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b9,r13,band_HIGH_PAIRED__Z2
        + 2.0281668543 aux_cap_helper__b9,r13,band_HIGH_PAIRED__condX1
        + 2.8394335959 aux_cap_helper__b9,r13,band_HIGH_PAIRED__condX2
        + 1.8253501688 aux_cap_helper__b9,r13,band_HIGH_PAIRED__condX3  = 0
 c2231: - aux_cap_helper__b6,r5,band_UNPAIRED__condX1 <= 0
 c2232: aux_cap_helper__b1,r12,band_LOW_PAIRED__Z1
        + aux_cap_helper__b1,r12,band_LOW_PAIRED__Z2  = 1
 c2233: - 6.60787655101763 sv_function_helpervar_b2,r13__Z2
        + sv_function_helpervar_b2,r13__condX2 <= 0
 c2234: 4.4504289328 aux_cap_helper__b2,r6,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b2,r6,band_LOW_PAIRED__condX1
        - 1.2715511236 aux_cap_helper__b2,r6,band_LOW_PAIRED__Z2
        - aux_cap_b2,r6,band_LOW_PAIRED  = 0
 c2235: - 0.158453086915011 sv_function_helpervar_b0,r0__Z1
        + sv_function_helpervar_b0,r0__condX1 <= 0
 c2236: - aux_cap_helper__b9,r10,band_UNPAIRED__condX2
        + aux_cap_helper__b9,r10,band_UNPAIRED__Z2 <= 0
 c2237: - 0.158453086953743 sv_function_helpervar_b9,r9__Z1
        + sv_function_helpervar_b9,r9__condX1 <= 0
 c2238: - aux_cap_helper__b8,r13,band_HIGH_PAIRED__condX1 <= 0
 c2239: aux_cap_helper__b0,r0,band_UNPAIRED__Z2
        + aux_cap_helper__b0,r0,band_UNPAIRED__Z1  = 1
 c2240: sv_function_helpervar_b2,r4__Z1 + sv_function_helpervar_b2,r4__Z2
        + sv_function_helpervar_b2,r4__Z3  = 1
 c2241: 2.8394335959 aux_cap_helper__b5,r12,band_HIGH_PAIRED__condX2
        + 1.8253501688 aux_cap_helper__b5,r12,band_HIGH_PAIRED__condX3
        + 2.0281668543 aux_cap_helper__b5,r12,band_HIGH_PAIRED__condX1
        + 1.21690011250031 aux_cap_helper__b5,r12,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b5,r12,band_HIGH_PAIRED__Z2
        - aux_cap_b5,r12,band_HIGH_PAIRED  = 0
 c2242: - aux_c_b5,r7 + sv_function_helpervar_b5,r7__condX1
        + sv_function_helpervar_b5,r7__condX2
        + sv_function_helpervar_b5,r7__condX3  = 0
 c2243: - sv_function_helpervar_b9,r2__condX1 <= 0
 c2244: aux_cap_helper__b1,r7,band_UNPAIRED__condX1
        - aux_cap_helper__b1,r7,band_UNPAIRED__Z1 <= 0
 c2245: - aux_cap_helper__b3,r3,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b3,r3,band_HIGH_PAIRED__condX1 <= 0
 c2246: - 0.158453086928019 sv_function_helpervar_b0,r13__Z1
        + sv_function_helpervar_b0,r13__condX1 <= 0
 c2247: aux_cap_b5,r5,band_HIGH_PAIRED - aux_c_b5,r5
        + aux_cap_b5,r5,band_LOW_PAIRED + aux_cap_b5,r5,band_UNPAIRED  = 0
 c2248: - aux_cap_helper__b2,r1,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b2,r1,band_HIGH_PAIRED__Z3 <= 0
 c2249: - aux_Omega_b9,r4 + 2837.68203858996 aux_sv_b9,r4  = 0
 c2250: 4.4504289328 aux_cap_helper__b5,r12,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b5,r12,band_LOW_PAIRED__condX1
        - aux_cap_b5,r12,band_LOW_PAIRED
        - 1.2715511236 aux_cap_helper__b5,r12,band_LOW_PAIRED__Z2  = 0
 c2251: - aux_cap_helper__b1,r5,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b1,r5,band_HIGH_PAIRED__Z2 <= 0
 c2252: - aux_cap_helper__b3,r11,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b3,r11,band_HIGH_PAIRED__Z2 <= 0
 c2253: - sv_function_helpervar_b4,r13__condX3
        + 6.26560685803533 sv_function_helpervar_b4,r13__Z3 <= 0
 c2254: aux_cap_helper__b8,r7,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b8,r7,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b8,r7,band_HIGH_PAIRED__Z1  = 1
 c2255: - sv_function_helpervar_b0,r8__condX1 <= 0
 c2256: aux_cap_helper__b2,r4,band_LOW_PAIRED__Z2
        - aux_cap_helper__b2,r4,band_LOW_PAIRED__condX2 <= 0
 c2257: - 2 aux_cap_helper__b1,r1,band_UNPAIRED__Z2
        + aux_cap_helper__b1,r1,band_UNPAIRED__condX2 <= 0
 c2258: - aux_cap_helper__b4,r5,band_HIGH_PAIRED__condX1 <= 0
 c2259: 178.34774530666 aux_sv_b0,r9 - aux_Omega_b0,r9  = 0
 c2260: aux_cap_b6,r3,band_LOW_PAIRED - aux_c_b6,r3
        + aux_cap_b6,r3,band_HIGH_PAIRED + aux_cap_b6,r3,band_UNPAIRED  = 0
 c2261: aux_cap_helper__b4,r9,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b4,r9,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b4,r9,band_HIGH_PAIRED__Z3  = 1
 c2262: - X__b8,r12,band_UNPAIRED
        + aux_cap_helper__b8,r12,band_UNPAIRED__condX2
        + aux_cap_helper__b8,r12,band_UNPAIRED__condX1  = 0
 c2263: - 2 aux_cap_helper__b0,r5,band_UNPAIRED__Z2
        + aux_cap_helper__b0,r5,band_UNPAIRED__condX2 <= 0
 c2264: - 15.8453086887446 sv_function_helpervar_b7,r4__Z3
        + sv_function_helpervar_b7,r4__condX3 <= 0
 c2265: sv_function_helpervar_b5,r9__condX1
        - 0.158453086965282 sv_function_helpervar_b5,r9__Z1 <= 0
 c2266: aux_cap_helper__b2,r2,band_LOW_PAIRED__condX1
        - aux_cap_helper__b2,r2,band_LOW_PAIRED__Z1 <= 0
 c2267: - aux_cap_helper__b1,r5,band_HIGH_PAIRED__condX1 <= 0
 c2268: GlobalBidder_W_i_9 - 14 GlobalBidder_hat{W}__i9,k1 >= -1
 c2269: aux_cap_helper__b9,r3,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b9,r3,band_HIGH_PAIRED__Z1 <= 0
 c2270: sv_function_helpervar_b4,r5__condX1
        - 0.158453086918754 sv_function_helpervar_b4,r5__Z1 <= 0
 c2271: - aux_cap_helper__b2,r4,band_LOW_PAIRED__condX1 <= 0
 c2272: sv_function_helpervar_b6,r7__condX3
        - 15.8453086887446 sv_function_helpervar_b6,r7__Z3 <= 0
 c2273: aux_cap_helper__b8,r13,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b8,r13,band_HIGH_PAIRED__condX2 <= 0
 c2274: aux_cap_helper__b7,r10,band_UNPAIRED__condX1
        - aux_cap_helper__b7,r10,band_UNPAIRED__Z1 <= 0
 c2275: aux_cap_helper__b6,r10,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b6,r10,band_HIGH_PAIRED__Z1 <= 0
 c2276: - aux_Omega_b5,r9 + 280.296801700826 aux_sv_b5,r9  = 0
 c2277: aux_cap_helper__b5,r6,band_UNPAIRED__Z1
        + aux_cap_helper__b5,r6,band_UNPAIRED__Z2  = 1
 c2278: aux_cap_b4,r11,band_HIGH_PAIRED + aux_cap_b4,r11,band_LOW_PAIRED
        + aux_cap_b4,r11,band_UNPAIRED - aux_c_b4,r11  = 0
 c2279: - aux_cap_helper__b2,r3,band_UNPAIRED__Z1
        + aux_cap_helper__b2,r3,band_UNPAIRED__condX1 <= 0
 c2280: aux_cap_helper__b6,r9,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b6,r9,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b6,r9,band_HIGH_PAIRED__Z3  = 1
 c2281: sv_function_helpervar_b6,r2__condX1
        + sv_function_helpervar_b6,r2__condX2
        + sv_function_helpervar_b6,r2__condX3 - aux_c_b6,r2  = 0
 c2282: 1995.2925271026 sv_function_helpervar_b9,r4__condX1
        + 37.9882751299 sv_function_helpervar_b9,r4__condX3
        + 73.1427030867 sv_function_helpervar_b9,r4__condX2
        + 304.570573103142 sv_function_helpervar_b9,r4__Z2
        + 569.027980650427 sv_function_helpervar_b9,r4__Z3 - aux_sv_b9,r4  = 0
 c2283: sv_function_helpervar_b5,r2__Z3 + sv_function_helpervar_b5,r2__Z2
        + sv_function_helpervar_b5,r2__Z1  = 1
 c2284: aux_cap_helper__b2,r2,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b2,r2,band_HIGH_PAIRED__Z1 <= 0
 c2285: - aux_cap_b1,r8,band_LOW_PAIRED
        - 1.2715511236 aux_cap_helper__b1,r8,band_LOW_PAIRED__Z2
        + 3.1788778092 aux_cap_helper__b1,r8,band_LOW_PAIRED__condX1
        + 4.4504289328 aux_cap_helper__b1,r8,band_LOW_PAIRED__condX2  = 0
 c2286: - aux_cap_helper__b0,r11,band_HIGH_PAIRED__condX1 <= 0
 c2287: - sv_function_helpervar_b3,r8__condX2
        + 0.15845308695304 sv_function_helpervar_b3,r8__Z2 <= 0
 c2288: - aux_Omega_b1,r6 + 45.1643170425842 aux_sv_b1,r6  = 0
 c2289: 783.051221032966 aux_sv_b8,r5 - aux_Omega_b8,r5  = 0
 c2290: aux_cap_helper__b4,r5,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b4,r5,band_HIGH_PAIRED__condX2 <= 0
 c2291: sv_function_helpervar_b5,r6__Z1 + sv_function_helpervar_b5,r6__Z2
        + sv_function_helpervar_b5,r6__Z3  = 1
 c2292: - sv_function_helpervar_b3,r10__condX1 <= 0
 c2293: aux_cap_helper__b4,r2,band_LOW_PAIRED__Z1
        + aux_cap_helper__b4,r2,band_LOW_PAIRED__Z2  = 1
 c2294: 153.718202239084 aux_sv_b0,r1 - aux_Omega_b0,r1  = 0
 c2295: - aux_cap_helper__b2,r7,band_LOW_PAIRED__condX1 <= 0
 c2296: aux_cap_helper__b5,r8,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b5,r8,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b5,r8,band_HIGH_PAIRED__Z2  = 1
 c2297: - aux_cap_helper__b0,r3,band_LOW_PAIRED__condX1 <= 0
 c2298: - 2 aux_cap_helper__b4,r13,band_UNPAIRED__Z2
        + aux_cap_helper__b4,r13,band_UNPAIRED__condX2 <= 0
 c2299: 2 aux_cap_helper__b5,r8,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b5,r8,band_HIGH_PAIRED__condX3 <= 0
 c2300: - sv_function_helpervar_b8,r0__condX3
        + 8.01571555507819 sv_function_helpervar_b8,r0__Z3 <= 0
 c2301: aux_cap_helper__b1,r10,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b1,r10,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b1,r10,band_HIGH_PAIRED__Z3  = 1
 c2302: aux_cap_helper__b2,r10,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b2,r10,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b2,r10,band_HIGH_PAIRED__condX1
        - X__b2,r10,band_HIGH_PAIRED  = 0
 c2303: aux_cap_b8,r12,band_UNPAIRED - aux_c_b8,r12
        + aux_cap_b8,r12,band_HIGH_PAIRED + aux_cap_b8,r12,band_LOW_PAIRED  = 0
 c2304: aux_cap_helper__b2,r2,band_UNPAIRED__Z1
        + aux_cap_helper__b2,r2,band_UNPAIRED__Z2  = 1
 c2305: aux_cap_helper__b6,r7,band_UNPAIRED__Z2
        + aux_cap_helper__b6,r7,band_UNPAIRED__Z1  = 1
 c2306: - 0.253841888 aux_cap_helper__b5,r12,band_UNPAIRED__Z2
        + 0.888446608 aux_cap_helper__b5,r12,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b5,r12,band_UNPAIRED__condX1
        - aux_cap_b5,r12,band_UNPAIRED  = 0
 c2307: 0.0102040816326531 X__b9,r9,band_LOW_PAIRED
        + 0.0102040816326531 X__b9,r9,band_UNPAIRED
        + 0.0102040816326531 X__b9,r9,band_HIGH_PAIRED
        - GlobalBidder_W_ir__b9,r9 <= 0
 c2308: - aux_cap_helper__b1,r2,band_LOW_PAIRED__condX1 <= 0
 c2309: sv_function_helpervar_b1,r9__condX3
        + sv_function_helpervar_b1,r9__condX1
        + sv_function_helpervar_b1,r9__condX2 - aux_c_b1,r9  = 0
 c2310: - sv_function_helpervar_b9,r12__condX1 <= 0
 c2311: - aux_cap_helper__b0,r11,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b0,r11,band_HIGH_PAIRED__Z2 <= 0
 c2312: - 3 aux_cap_helper__b5,r5,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b5,r5,band_HIGH_PAIRED__condX3 <= 0
 c2313: - aux_cap_helper__b7,r11,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b7,r11,band_HIGH_PAIRED__condX1 <= 0
 c2314: - aux_cap_helper__b8,r1,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b8,r1,band_HIGH_PAIRED__Z3 <= 0
 c2315: - aux_sv_b0,r5 + 163.7425744011 sv_function_helpervar_b0,r5__condX1
        + 2.7658052108 sv_function_helpervar_b0,r5__condX3
        + 7.0097068589 sv_function_helpervar_b0,r5__condX2
        + 24.8348066797091 sv_function_helpervar_b0,r5__Z2
        + 52.2694677354168 sv_function_helpervar_b0,r5__Z3  = 0
 c2316: - sv_function_helpervar_b8,r6__condX3
        + 6.10570058921676 sv_function_helpervar_b8,r6__Z3 <= 0
 c2317: - 15.8453086887446 sv_function_helpervar_b3,r0__Z3
        + sv_function_helpervar_b3,r0__condX3 <= 0
 c2318: - aux_c_b4,r11 + sv_function_helpervar_b4,r11__condX2
        + sv_function_helpervar_b4,r11__condX1
        + sv_function_helpervar_b4,r11__condX3  = 0
 c2319: aux_cap_helper__b2,r5,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b2,r5,band_HIGH_PAIRED__Z3 <= 0
 c2320: sv_function_helpervar_b2,r4__condX1
        - 0.158453086899697 sv_function_helpervar_b2,r4__Z1 <= 0
 c2321: sv_function_helpervar_b9,r9__Z3 + sv_function_helpervar_b9,r9__Z2
        + sv_function_helpervar_b9,r9__Z1  = 1
 c2322: - aux_cap_helper__b6,r4,band_HIGH_PAIRED__condX1 <= 0
 c2323: - aux_cap_helper__b0,r3,band_LOW_PAIRED__condX2
        + aux_cap_helper__b0,r3,band_LOW_PAIRED__Z2 <= 0
 c2324: - 1.2715511236 aux_cap_helper__b6,r3,band_LOW_PAIRED__Z2
        - aux_cap_b6,r3,band_LOW_PAIRED
        + 3.1788778092 aux_cap_helper__b6,r3,band_LOW_PAIRED__condX1
        + 4.4504289328 aux_cap_helper__b6,r3,band_LOW_PAIRED__condX2  = 0
 c2325: aux_cap_helper__b0,r5,band_UNPAIRED__condX1
        + aux_cap_helper__b0,r5,band_UNPAIRED__condX2 - X__b0,r5,band_UNPAIRED
         = 0
 c2326: - 2 aux_cap_helper__b8,r7,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b8,r7,band_HIGH_PAIRED__condX2 <= 0
 c2327: - sv_function_helpervar_b2,r3__condX2
        + 0.158453086903001 sv_function_helpervar_b2,r3__Z2 <= 0
 c2328: - 0.158453086929204 sv_function_helpervar_b5,r11__Z1
        + sv_function_helpervar_b5,r11__condX1 <= 0
 c2329: aux_cap_helper__b1,r4,band_LOW_PAIRED__Z2
        + aux_cap_helper__b1,r4,band_LOW_PAIRED__Z1  = 1
 c2330: - aux_cap_b1,r4,band_LOW_PAIRED
        + 4.4504289328 aux_cap_helper__b1,r4,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b1,r4,band_LOW_PAIRED__condX1
        - 1.2715511236 aux_cap_helper__b1,r4,band_LOW_PAIRED__Z2  = 0
 c2331: aux_cap_helper__b0,r6,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b0,r6,band_HIGH_PAIRED__Z3 <= 0
 c2332: 295.185321133132 sv_function_helpervar_b4,r0__Z3
        + 133.408237358556 sv_function_helpervar_b4,r0__Z2
        + 41.4612591645 sv_function_helpervar_b4,r0__condX2
        + 883.4028076482 sv_function_helpervar_b4,r0__condX1
        + 14.0894290424 sv_function_helpervar_b4,r0__condX3 - aux_sv_b4,r0  = 0
 c2333: - 0.158453086996438 sv_function_helpervar_b0,r6__Z1
        + sv_function_helpervar_b0,r6__condX1 <= 0
 c2334: - 15.8453086887446 sv_function_helpervar_b6,r10__Z3
        + sv_function_helpervar_b6,r10__condX3 <= 0
 c2335: aux_cap_helper__b7,r0,band_LOW_PAIRED__Z1
        + aux_cap_helper__b7,r0,band_LOW_PAIRED__Z2  = 1
 c2336: aux_cap_helper__b3,r11,band_UNPAIRED__Z2
        - aux_cap_helper__b3,r11,band_UNPAIRED__condX2 <= 0
 c2337: sv_function_helpervar_b3,r8__condX3
        + sv_function_helpervar_b3,r8__condX2
        + sv_function_helpervar_b3,r8__condX1 - aux_c_b3,r8  = 0
 c2338: sv_function_helpervar_b1,r1__condX2
        + sv_function_helpervar_b1,r1__condX1
        + sv_function_helpervar_b1,r1__condX3 - aux_c_b1,r1  = 0
 c2339: X__b7,r13,band_HIGH_PAIRED + X__b7,r13,band_UNPAIRED
        + X__b7,r13,band_LOW_PAIRED - GlobalBidder_W_ir__b7,r13 >= 0
 c2340: 2.0281668543 aux_cap_helper__b3,r10,band_HIGH_PAIRED__condX1
        + 2.8394335959 aux_cap_helper__b3,r10,band_HIGH_PAIRED__condX2
        + 1.8253501688 aux_cap_helper__b3,r10,band_HIGH_PAIRED__condX3
        - aux_cap_b3,r10,band_HIGH_PAIRED
        - 0.8112667416 aux_cap_helper__b3,r10,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b3,r10,band_HIGH_PAIRED__Z3  = 0
 c2341: 1.21690011250031 aux_cap_helper__b4,r5,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b4,r5,band_HIGH_PAIRED__Z2
        - aux_cap_b4,r5,band_HIGH_PAIRED
        + 1.8253501688 aux_cap_helper__b4,r5,band_HIGH_PAIRED__condX3
        + 2.0281668543 aux_cap_helper__b4,r5,band_HIGH_PAIRED__condX1
        + 2.8394335959 aux_cap_helper__b4,r5,band_HIGH_PAIRED__condX2  = 0
 c2342: - X__b6,r0,band_UNPAIRED + aux_cap_helper__b6,r0,band_UNPAIRED__condX2
        + aux_cap_helper__b6,r0,band_UNPAIRED__condX1  = 0
 c2343: aux_cap_helper__b1,r6,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b1,r6,band_HIGH_PAIRED__Z2 <= 0
 c2344: aux_cap_helper__b7,r6,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b7,r6,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b7,r6,band_HIGH_PAIRED__condX1
        - X__b7,r6,band_HIGH_PAIRED  = 0
 c2345: 6.1718003564743 sv_function_helpervar_b6,r0__Z3
        - sv_function_helpervar_b6,r0__condX3 <= 0
 c2346: - GlobalBidder_W_ir__b9,r6 + X__b9,r6,band_UNPAIRED
        + X__b9,r6,band_HIGH_PAIRED + X__b9,r6,band_LOW_PAIRED >= 0
 c2347: - aux_cap_helper__b9,r4,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b9,r4,band_HIGH_PAIRED__Z2 <= 0
 c2348: sv_function_helpervar_b8,r0__condX3
        - 15.8453086887446 sv_function_helpervar_b8,r0__Z3 <= 0
 c2349: 2 aux_cap_helper__b3,r9,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b3,r9,band_HIGH_PAIRED__condX3 <= 0
 c2350: - aux_cap_helper__b8,r12,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b8,r12,band_HIGH_PAIRED__condX1 <= 0
 c2351: 7.02238717159649 sv_function_helpervar_b8,r1__Z3
        - sv_function_helpervar_b8,r1__condX3 <= 0
 c2352: - 0.8112667416 aux_cap_helper__b1,r8,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b1,r8,band_HIGH_PAIRED__Z3
        + 2.0281668543 aux_cap_helper__b1,r8,band_HIGH_PAIRED__condX1
        + 2.8394335959 aux_cap_helper__b1,r8,band_HIGH_PAIRED__condX2
        + 1.8253501688 aux_cap_helper__b1,r8,band_HIGH_PAIRED__condX3
        - aux_cap_b1,r8,band_HIGH_PAIRED  = 0
 c2353: - 2 aux_cap_helper__b1,r10,band_UNPAIRED__Z2
        + aux_cap_helper__b1,r10,band_UNPAIRED__condX2 <= 0
 c2354: - aux_cap_helper__b6,r4,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b6,r4,band_HIGH_PAIRED__Z2 <= 0
 c2355: sv_function_helpervar_b8,r11__Z1 + sv_function_helpervar_b8,r11__Z2
        + sv_function_helpervar_b8,r11__Z3  = 1
 c2356: aux_cap_helper__b4,r10,band_UNPAIRED__condX2
        + aux_cap_helper__b4,r10,band_UNPAIRED__condX1
        - X__b4,r10,band_UNPAIRED  = 0
 c2357: 0.158453086896389 sv_function_helpervar_b7,r0__Z2
        - sv_function_helpervar_b7,r0__condX2 <= 0
 c2358: - 1.2715511236 aux_cap_helper__b8,r7,band_LOW_PAIRED__Z2
        - aux_cap_b8,r7,band_LOW_PAIRED
        + 4.4504289328 aux_cap_helper__b8,r7,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b8,r7,band_LOW_PAIRED__condX1  = 0
 c2359: - aux_cap_helper__b9,r4,band_HIGH_PAIRED__condX1 <= 0
 c2360: 2 aux_cap_helper__b0,r9,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b0,r9,band_HIGH_PAIRED__condX3 <= 0
 c2361: sv_function_helpervar_b1,r4__condX2
        - 5.93643512983155 sv_function_helpervar_b1,r4__Z2 <= 0
 c2362: 0.888446608 aux_cap_helper__b3,r12,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b3,r12,band_UNPAIRED__condX1
        - 0.253841888 aux_cap_helper__b3,r12,band_UNPAIRED__Z2
        - aux_cap_b3,r12,band_UNPAIRED  = 0
 c2363: - X__b6,r12,band_HIGH_PAIRED
        + aux_cap_helper__b6,r12,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b6,r12,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b6,r12,band_HIGH_PAIRED__condX3  = 0
 c2364: - 15.8453086887446 sv_function_helpervar_b8,r9__Z3
        + sv_function_helpervar_b8,r9__condX3 <= 0
 c2365: - aux_Omega_b2,r9 + 385.753606031818 aux_sv_b2,r9  = 0
 c2366: 0.888446608 aux_cap_helper__b0,r0,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b0,r0,band_UNPAIRED__condX1
        - aux_cap_b0,r0,band_UNPAIRED
        - 0.253841888 aux_cap_helper__b0,r0,band_UNPAIRED__Z2  = 0
 c2367: 24.7277380836865 sv_function_helpervar_b0,r11__Z2
        + 54.715114730046 sv_function_helpervar_b0,r11__Z3
        + 7.6854185038 sv_function_helpervar_b0,r11__condX2
        + 163.7425743958 sv_function_helpervar_b0,r11__condX1
        + 2.6114600325 sv_function_helpervar_b0,r11__condX3 - aux_sv_b0,r11
         = 0
 c2368: 2 aux_cap_helper__b1,r0,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b1,r0,band_HIGH_PAIRED__condX3 <= 0
 c2369: aux_cap_helper__b6,r8,band_UNPAIRED__condX1
        - aux_cap_helper__b6,r8,band_UNPAIRED__Z1 <= 0
 c2370: aux_cap_helper__b7,r8,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b7,r8,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b7,r8,band_HIGH_PAIRED__Z3  = 1
 c2371: - 6.52004106046921 sv_function_helpervar_b0,r13__Z2
        + sv_function_helpervar_b0,r13__condX2 <= 0
 c2372: 0.15845308694837 sv_function_helpervar_b0,r4__Z2
        - sv_function_helpervar_b0,r4__condX2 <= 0
 c2373: - aux_cap_helper__b3,r4,band_HIGH_PAIRED__condX1 <= 0
 c2374: 0.63460472 aux_cap_helper__b1,r8,band_UNPAIRED__condX1
        + 0.888446608 aux_cap_helper__b1,r8,band_UNPAIRED__condX2
        - aux_cap_b1,r8,band_UNPAIRED
        - 0.253841888 aux_cap_helper__b1,r8,band_UNPAIRED__Z2  = 0
 c2375: - aux_Omega_b2,r1 + 255.83710108514 aux_sv_b2,r1  = 0
 c2376: sv_function_helpervar_b3,r9__condX3
        - 15.8453086887446 sv_function_helpervar_b3,r9__Z3 <= 0
 c2377: - aux_cap_helper__b5,r12,band_UNPAIRED__Z1
        + aux_cap_helper__b5,r12,band_UNPAIRED__condX1 <= 0
 c2378: - 2 aux_cap_helper__b9,r6,band_LOW_PAIRED__Z2
        + aux_cap_helper__b9,r6,band_LOW_PAIRED__condX2 <= 0
 c2379: 4.4504289328 aux_cap_helper__b7,r10,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b7,r10,band_LOW_PAIRED__condX1
        - 1.2715511236 aux_cap_helper__b7,r10,band_LOW_PAIRED__Z2
        - aux_cap_b7,r10,band_LOW_PAIRED  = 0
 c2380: aux_cap_helper__b7,r8,band_UNPAIRED__Z2
        + aux_cap_helper__b7,r8,band_UNPAIRED__Z1  = 1
 c2381: aux_cap_helper__b3,r13,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b3,r13,band_HIGH_PAIRED__Z3 <= 0
 c2382: - aux_c_b2,r0 + aux_cap_b2,r0,band_UNPAIRED
        + aux_cap_b2,r0,band_LOW_PAIRED + aux_cap_b2,r0,band_HIGH_PAIRED  = 0
 c2383: - aux_cap_helper__b4,r6,band_LOW_PAIRED__Z1
        + aux_cap_helper__b4,r6,band_LOW_PAIRED__condX1 <= 0
 c2384: - aux_cap_helper__b5,r8,band_LOW_PAIRED__Z1
        + aux_cap_helper__b5,r8,band_LOW_PAIRED__condX1 <= 0
 c2385: aux_cap_b3,r11,band_UNPAIRED - aux_c_b3,r11
        + aux_cap_b3,r11,band_HIGH_PAIRED + aux_cap_b3,r11,band_LOW_PAIRED  = 0
 c2386: X__b8,r8,band_HIGH_PAIRED + X__b8,r8,band_LOW_PAIRED
        - GlobalBidder_W_ir__b8,r8 + X__b8,r8,band_UNPAIRED >= 0
 c2387: aux_cap_helper__b9,r13,band_LOW_PAIRED__Z2
        + aux_cap_helper__b9,r13,band_LOW_PAIRED__Z1  = 1
 c2388: - sv_function_helpervar_b6,r4__condX1 <= 0
 c2389: sv_function_helpervar_b0,r2__condX3
        - 15.8453086887446 sv_function_helpervar_b0,r2__Z3 <= 0
 c2390: - aux_cap_helper__b8,r2,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b8,r2,band_HIGH_PAIRED__condX1 <= 0
 c2391: - 0.253841888 aux_cap_helper__b9,r9,band_UNPAIRED__Z2
        + 0.63460472 aux_cap_helper__b9,r9,band_UNPAIRED__condX1
        + 0.888446608 aux_cap_helper__b9,r9,band_UNPAIRED__condX2
        - aux_cap_b9,r9,band_UNPAIRED  = 0
 c2392: aux_cap_helper__b5,r5,band_UNPAIRED__condX1
        - aux_cap_helper__b5,r5,band_UNPAIRED__Z1 <= 0
 c2393: - aux_cap_helper__b3,r11,band_UNPAIRED__condX1 <= 0
 c2394: 13.9952556763 sv_function_helpervar_b4,r8__condX3
        + 41.948812298 sv_function_helpervar_b4,r8__condX2
        + 883.402807655 sv_function_helpervar_b4,r8__condX1
        + 133.330983059575 sv_function_helpervar_b4,r8__Z2
        + 296.677527189472 sv_function_helpervar_b4,r8__Z3 - aux_sv_b4,r8  = 0
 c2395: aux_cap_helper__b0,r2,band_UNPAIRED__condX1
        + aux_cap_helper__b0,r2,band_UNPAIRED__condX2 - X__b0,r2,band_UNPAIRED
         = 0
 c2396: aux_cap_helper__b8,r9,band_LOW_PAIRED__Z2
        + aux_cap_helper__b8,r9,band_LOW_PAIRED__Z1  = 1
 c2397: - X__b3,r4,band_UNPAIRED + aux_cap_helper__b3,r4,band_UNPAIRED__condX1
        + aux_cap_helper__b3,r4,band_UNPAIRED__condX2  = 0
 c2398: 6.29672555527569 sv_function_helpervar_b4,r1__Z3
        - sv_function_helpervar_b4,r1__condX3 <= 0
 c2399: - aux_cap_helper__b5,r2,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b5,r2,band_HIGH_PAIRED__condX1 <= 0
 c2400: X__b8,r4,band_HIGH_PAIRED + X__b4,r4,band_HIGH_PAIRED
        + X__b7,r4,band_HIGH_PAIRED + X__b9,r4,band_HIGH_PAIRED
        + X__b6,r4,band_HIGH_PAIRED + X__b5,r4,band_HIGH_PAIRED
        + X__b0,r4,band_HIGH_PAIRED + X__b1,r4,band_HIGH_PAIRED
        + X__b3,r4,band_HIGH_PAIRED + X__b2,r4,band_HIGH_PAIRED <= 3
 c2401: - aux_Omega_b2,r8 + 143.831209447361 aux_sv_b2,r8  = 0
 c2402: 0.63460472 aux_cap_helper__b8,r1,band_UNPAIRED__condX1
        + 0.888446608 aux_cap_helper__b8,r1,band_UNPAIRED__condX2
        - aux_cap_b8,r1,band_UNPAIRED
        - 0.253841888 aux_cap_helper__b8,r1,band_UNPAIRED__Z2  = 0
 c2403: sv_function_helpervar_b2,r0__Z1 + sv_function_helpervar_b2,r0__Z2
        + sv_function_helpervar_b2,r0__Z3  = 1
 c2404: - aux_cap_helper__b4,r9,band_UNPAIRED__Z1
        + aux_cap_helper__b4,r9,band_UNPAIRED__condX1 <= 0
 c2405: - X__b0,r4,band_LOW_PAIRED
        + aux_cap_helper__b0,r4,band_LOW_PAIRED__condX1
        + aux_cap_helper__b0,r4,band_LOW_PAIRED__condX2  = 0
 c2406: aux_cap_helper__b3,r8,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b3,r8,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b3,r8,band_HIGH_PAIRED__Z1  = 1
 c2407: sv_function_helpervar_b3,r13__Z2 + sv_function_helpervar_b3,r13__Z3
        + sv_function_helpervar_b3,r13__Z1  = 1
 c2408: - sv_function_helpervar_b3,r6__condX1 <= 0
 c2409: aux_cap_helper__b0,r8,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b0,r8,band_LOW_PAIRED__Z2 <= 0
 c2410: aux_cap_helper__b0,r8,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b0,r8,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b0,r8,band_HIGH_PAIRED__condX1
        - X__b0,r8,band_HIGH_PAIRED  = 0
 c2411: aux_cap_helper__b1,r13,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b1,r13,band_HIGH_PAIRED__Z2 <= 0
 c2412: - aux_cap_helper__b0,r8,band_UNPAIRED__condX1 <= 0
 c2413: - 1.2715511236 aux_cap_helper__b0,r2,band_LOW_PAIRED__Z2
        - aux_cap_b0,r2,band_LOW_PAIRED
        + 4.4504289328 aux_cap_helper__b0,r2,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b0,r2,band_LOW_PAIRED__condX1  = 0
 c2414: aux_cap_helper__b3,r9,band_LOW_PAIRED__Z2
        - aux_cap_helper__b3,r9,band_LOW_PAIRED__condX2 <= 0
 c2415: 5.56014354927208 sv_function_helpervar_b2,r5__Z3
        - sv_function_helpervar_b2,r5__condX3 <= 0
 c2416: - aux_cap_helper__b2,r7,band_LOW_PAIRED__condX2
        + aux_cap_helper__b2,r7,band_LOW_PAIRED__Z2 <= 0
 c2417: aux_cap_helper__b4,r13,band_UNPAIRED__condX2
        + aux_cap_helper__b4,r13,band_UNPAIRED__condX1
        - X__b4,r13,band_UNPAIRED  = 0
 c2418: aux_cap_helper__b0,r13,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b0,r13,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b0,r13,band_HIGH_PAIRED__Z3  = 1
 c2419: - aux_Omega_b9,r1 + 353.143105067263 aux_sv_b9,r1  = 0
 c2420: - X__b2,r2,band_LOW_PAIRED
        + aux_cap_helper__b2,r2,band_LOW_PAIRED__condX1
        + aux_cap_helper__b2,r2,band_LOW_PAIRED__condX2  = 0
 c2421: sv_function_helpervar_b3,r0__condX1
        + sv_function_helpervar_b3,r0__condX2
        + sv_function_helpervar_b3,r0__condX3 - aux_c_b3,r0  = 0
 c2422: 0.158453086891007 sv_function_helpervar_b8,r5__Z2
        - sv_function_helpervar_b8,r5__condX2 <= 0
 c2423: - 15.8453086887446 sv_function_helpervar_b5,r2__Z3
        + sv_function_helpervar_b5,r2__condX3 <= 0
 c2424: X__b8,r10,band_LOW_PAIRED + X__b1,r10,band_LOW_PAIRED
        + X__b9,r10,band_LOW_PAIRED + X__b2,r10,band_LOW_PAIRED
        + X__b0,r10,band_LOW_PAIRED + X__b3,r10,band_LOW_PAIRED
        + X__b4,r10,band_LOW_PAIRED + X__b7,r10,band_LOW_PAIRED
        + X__b5,r10,band_LOW_PAIRED + X__b6,r10,band_LOW_PAIRED <= 2
 c2425: - X__b4,r0,band_LOW_PAIRED
        + aux_cap_helper__b4,r0,band_LOW_PAIRED__condX1
        + aux_cap_helper__b4,r0,band_LOW_PAIRED__condX2  = 0
 c2426: 4.4504289328 aux_cap_helper__b7,r5,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b7,r5,band_LOW_PAIRED__condX1
        - 1.2715511236 aux_cap_helper__b7,r5,band_LOW_PAIRED__Z2
        - aux_cap_b7,r5,band_LOW_PAIRED  = 0
 c2427: - aux_sv_b8,r4 + 32.4564252682 sv_function_helpervar_b8,r4__condX3
        + 1636.3575438917 sv_function_helpervar_b8,r4__condX1
        + 57.3836909814 sv_function_helpervar_b8,r4__condX2
        + 446.036086151013 sv_function_helpervar_b8,r4__Z3
        + 250.193281158827 sv_function_helpervar_b8,r4__Z2  = 0
 c2428: - aux_cap_helper__b0,r11,band_UNPAIRED__Z1
        + aux_cap_helper__b0,r11,band_UNPAIRED__condX1 <= 0
 c2429: - aux_Omega_b6,r4 + 697.398555180679 aux_sv_b6,r4  = 0
 c2430: - aux_cap_helper__b2,r0,band_UNPAIRED__condX2
        + aux_cap_helper__b2,r0,band_UNPAIRED__Z2 <= 0
 c2431: - aux_c_b5,r13 + sv_function_helpervar_b5,r13__condX1
        + sv_function_helpervar_b5,r13__condX3
        + sv_function_helpervar_b5,r13__condX2  = 0
 c2432: aux_cap_helper__b1,r4,band_UNPAIRED__Z2
        - aux_cap_helper__b1,r4,band_UNPAIRED__condX2 <= 0
 c2433: - aux_cap_helper__b4,r12,band_HIGH_PAIRED__condX1 <= 0
 c2434: aux_cap_helper__b7,r13,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b7,r13,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b7,r13,band_HIGH_PAIRED__Z3  = 1
 c2435: sv_function_helpervar_b8,r12__condX2
        - 6.2737382343784 sv_function_helpervar_b8,r12__Z2 <= 0
 c2436: - aux_Omega_b6,r0 + 131.623839580633 aux_sv_b6,r0  = 0
 c2437: - aux_cap_helper__b2,r0,band_UNPAIRED__condX1 <= 0
 c2438: aux_cap_helper__b8,r8,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b8,r8,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b8,r8,band_HIGH_PAIRED__condX2
        - X__b8,r8,band_HIGH_PAIRED  = 0
 c2439: aux_cap_helper__b0,r8,band_UNPAIRED__Z2
        - aux_cap_helper__b0,r8,band_UNPAIRED__condX2 <= 0
 c2440: - aux_cap_helper__b1,r4,band_UNPAIRED__condX1 <= 0
 c2441: 6.5034000540409 sv_function_helpervar_b1,r8__Z3
        - sv_function_helpervar_b1,r8__condX3 <= 0
 c2442: 2.0281668543 aux_cap_helper__b6,r2,band_HIGH_PAIRED__condX1
        + 1.8253501688 aux_cap_helper__b6,r2,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b6,r2,band_HIGH_PAIRED__condX2
        - aux_cap_b6,r2,band_HIGH_PAIRED
        - 0.8112667416 aux_cap_helper__b6,r2,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b6,r2,band_HIGH_PAIRED__Z3  = 0
 c2443: - aux_c_b8,r9 + sv_function_helpervar_b8,r9__condX1
        + sv_function_helpervar_b8,r9__condX2
        + sv_function_helpervar_b8,r9__condX3  = 0
 c2444: - X__b9,r2,band_UNPAIRED + aux_cap_helper__b9,r2,band_UNPAIRED__condX2
        + aux_cap_helper__b9,r2,band_UNPAIRED__condX1  = 0
 c2445: aux_cap_helper__b9,r6,band_LOW_PAIRED__condX1
        + aux_cap_helper__b9,r6,band_LOW_PAIRED__condX2
        - X__b9,r6,band_LOW_PAIRED  = 0
 c2446: sv_function_helpervar_b9,r5__Z3 + sv_function_helpervar_b9,r5__Z1
        + sv_function_helpervar_b9,r5__Z2  = 1
 c2447: aux_cap_b0,r3,band_UNPAIRED + aux_cap_b0,r3,band_LOW_PAIRED
        + aux_cap_b0,r3,band_HIGH_PAIRED - aux_c_b0,r3  = 0
 c2448: - aux_cap_helper__b5,r10,band_HIGH_PAIRED__condX1 <= 0
 c2449: - aux_cap_helper__b9,r5,band_UNPAIRED__condX2
        + aux_cap_helper__b9,r5,band_UNPAIRED__Z2 <= 0
 c2450: - 2 aux_cap_helper__b2,r9,band_UNPAIRED__Z2
        + aux_cap_helper__b2,r9,band_UNPAIRED__condX2 <= 0
 c2451: 0.158453086983578 sv_function_helpervar_b2,r5__Z2
        - sv_function_helpervar_b2,r5__condX2 <= 0
 c2452: - 2 aux_cap_helper__b8,r12,band_UNPAIRED__Z2
        + aux_cap_helper__b8,r12,band_UNPAIRED__condX2 <= 0
 c2453: - aux_Omega_b2,r7 + 1716.31844816029 aux_sv_b2,r7  = 0
 c2454: - 0.8112667416 aux_cap_helper__b5,r7,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b5,r7,band_HIGH_PAIRED__Z3
        + 2.8394335959 aux_cap_helper__b5,r7,band_HIGH_PAIRED__condX2
        + 1.8253501688 aux_cap_helper__b5,r7,band_HIGH_PAIRED__condX3
        + 2.0281668543 aux_cap_helper__b5,r7,band_HIGH_PAIRED__condX1
        - aux_cap_b5,r7,band_HIGH_PAIRED  = 0
 c2455: aux_cap_helper__b5,r9,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b5,r9,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b5,r9,band_HIGH_PAIRED__Z1  = 1
 c2456: - X__b8,r13,band_UNPAIRED
        + aux_cap_helper__b8,r13,band_UNPAIRED__condX1
        + aux_cap_helper__b8,r13,band_UNPAIRED__condX2  = 0
 c2457: - sv_function_helpervar_b9,r10__condX2
        + 0.158453086914684 sv_function_helpervar_b9,r10__Z2 <= 0
 c2458: aux_cap_helper__b3,r3,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b3,r3,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b3,r3,band_HIGH_PAIRED__Z1  = 1
 c2459: - aux_cap_helper__b0,r4,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b0,r4,band_HIGH_PAIRED__Z2 <= 0
 c2460: - aux_cap_helper__b4,r6,band_UNPAIRED__condX2
        + aux_cap_helper__b4,r6,band_UNPAIRED__Z2 <= 0
 c2461: - X__b6,r3,band_UNPAIRED + aux_cap_helper__b6,r3,band_UNPAIRED__condX2
        + aux_cap_helper__b6,r3,band_UNPAIRED__condX1  = 0
 c2462: aux_cap_helper__b6,r7,band_LOW_PAIRED__condX2
        + aux_cap_helper__b6,r7,band_LOW_PAIRED__condX1
        - X__b6,r7,band_LOW_PAIRED  = 0
 c2463: - 3 aux_cap_helper__b4,r12,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b4,r12,band_HIGH_PAIRED__condX3 <= 0
 c2464: X__b7,r10,band_HIGH_PAIRED + X__b7,r10,band_UNPAIRED
        + X__b7,r10,band_LOW_PAIRED - GlobalBidder_W_ir__b7,r10 >= 0
 c2465: - X__b3,r3,band_HIGH_PAIRED
        + aux_cap_helper__b3,r3,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b3,r3,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b3,r3,band_HIGH_PAIRED__condX1  = 0
 c2466: - 3 aux_cap_helper__b6,r3,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b6,r3,band_HIGH_PAIRED__condX3 <= 0
 c2467: - aux_cap_b9,r8,band_HIGH_PAIRED
        + 1.21690011250031 aux_cap_helper__b9,r8,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b9,r8,band_HIGH_PAIRED__Z2
        + 2.8394335959 aux_cap_helper__b9,r8,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b9,r8,band_HIGH_PAIRED__condX1
        + 1.8253501688 aux_cap_helper__b9,r8,band_HIGH_PAIRED__condX3  = 0
 c2468: - aux_cap_helper__b5,r6,band_HIGH_PAIRED__condX1 <= 0
 c2469: - 15.8453086887446 sv_function_helpervar_b0,r7__Z3
        + sv_function_helpervar_b0,r7__condX3 <= 0
 c2470: - aux_sv_b8,r3 + 482.061490839459 sv_function_helpervar_b8,r3__Z3
        + 249.422235627918 sv_function_helpervar_b8,r3__Z2
        + 30.1828561383 sv_function_helpervar_b8,r3__condX3
        + 62.2497718309 sv_function_helpervar_b8,r3__condX2
        + 1636.3575440395 sv_function_helpervar_b8,r3__condX1  = 0
 c2471: - aux_c_b0,r12 + aux_cap_b0,r12,band_LOW_PAIRED
        + aux_cap_b0,r12,band_UNPAIRED + aux_cap_b0,r12,band_HIGH_PAIRED  = 0
 c2472: aux_cap_helper__b3,r7,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b3,r7,band_HIGH_PAIRED__Z3 <= 0
 c2473: - aux_cap_helper__b8,r0,band_UNPAIRED__condX1 <= 0
 c2474: - aux_Omega_b9,r6 + 110.733810175324 aux_sv_b9,r6  = 0
 c2475: - aux_cap_helper__b6,r9,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b6,r9,band_HIGH_PAIRED__condX1 <= 0
 c2476: sv_function_helpervar_b6,r6__condX1
        - 0.158453086913347 sv_function_helpervar_b6,r6__Z1 <= 0
 c2477: aux_Omega_b9,r1 + aux_Omega_b9,r0 + aux_Omega_b9,r3 + aux_Omega_b9,r2
        + aux_Omega_b9,r5 + aux_Omega_b9,r4 + aux_Omega_b9,r7 + aux_Omega_b9,r6
        + aux_Omega_b9,r9 + aux_Omega_b9,r8 - GlobalBidder_PSI__i9,k2
        + aux_Omega_b9,r13 + aux_Omega_b9,r10 + aux_Omega_b9,r11
        + aux_Omega_b9,r12 + 20668026.8748422 GlobalBidder_hat{W}__i9,k2
        <= 20668026.8748422
 c2478: - aux_c_b7,r12 + sv_function_helpervar_b7,r12__condX3
        + sv_function_helpervar_b7,r12__condX2
        + sv_function_helpervar_b7,r12__condX1  = 0
 c2479: 1.8253501688 aux_cap_helper__b1,r6,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b1,r6,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b1,r6,band_HIGH_PAIRED__condX1
        + 1.21690011250031 aux_cap_helper__b1,r6,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b1,r6,band_HIGH_PAIRED__Z2
        - aux_cap_b1,r6,band_HIGH_PAIRED  = 0
 c2480: - aux_c_b0,r12 + sv_function_helpervar_b0,r12__condX2
        + sv_function_helpervar_b0,r12__condX3
        + sv_function_helpervar_b0,r12__condX1  = 0
 c2481: - 2 aux_cap_helper__b3,r9,band_LOW_PAIRED__Z2
        + aux_cap_helper__b3,r9,band_LOW_PAIRED__condX2 <= 0
 c2482: - aux_cap_helper__b3,r1,band_UNPAIRED__condX1 <= 0
 c2483: aux_cap_helper__b7,r8,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b7,r8,band_UNPAIRED__Z2 <= 0
 c2484: aux_cap_helper__b5,r0,band_LOW_PAIRED__Z2
        + aux_cap_helper__b5,r0,band_LOW_PAIRED__Z1  = 1
 c2485: 2.0281668543 aux_cap_helper__b7,r7,band_HIGH_PAIRED__condX1
        + 2.8394335959 aux_cap_helper__b7,r7,band_HIGH_PAIRED__condX2
        + 1.8253501688 aux_cap_helper__b7,r7,band_HIGH_PAIRED__condX3
        - aux_cap_b7,r7,band_HIGH_PAIRED
        + 1.21690011250031 aux_cap_helper__b7,r7,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b7,r7,band_HIGH_PAIRED__Z2  = 0
 c2486: - aux_Omega_b5,r7 + 2731.16687313042 aux_sv_b5,r7  = 0
 c2487: - aux_cap_helper__b1,r4,band_LOW_PAIRED__condX2
        + aux_cap_helper__b1,r4,band_LOW_PAIRED__Z2 <= 0
 c2488: - 2 aux_cap_helper__b2,r3,band_LOW_PAIRED__Z2
        + aux_cap_helper__b2,r3,band_LOW_PAIRED__condX2 <= 0
 c2489: aux_cap_helper__b0,r12,band_LOW_PAIRED__condX1
        + aux_cap_helper__b0,r12,band_LOW_PAIRED__condX2
        - X__b0,r12,band_LOW_PAIRED  = 0
 c2490: 3.1788778092 aux_cap_helper__b0,r12,band_LOW_PAIRED__condX1
        + 4.4504289328 aux_cap_helper__b0,r12,band_LOW_PAIRED__condX2
        - aux_cap_b0,r12,band_LOW_PAIRED
        - 1.2715511236 aux_cap_helper__b0,r12,band_LOW_PAIRED__Z2  = 0
 c2491: - aux_Omega_b7,r6 + 93.6463378727092 aux_sv_b7,r6  = 0
 c2492: aux_cap_helper__b7,r11,band_LOW_PAIRED__Z2
        + aux_cap_helper__b7,r11,band_LOW_PAIRED__Z1  = 1
 c2493: X__b0,r9,band_UNPAIRED + X__b9,r9,band_UNPAIRED
        + X__b1,r9,band_UNPAIRED + X__b6,r9,band_UNPAIRED
        + X__b5,r9,band_UNPAIRED + X__b7,r9,band_UNPAIRED
        + X__b4,r9,band_UNPAIRED + X__b2,r9,band_UNPAIRED
        + X__b8,r9,band_UNPAIRED + X__b3,r9,band_UNPAIRED <= 2
 c2494: 0.0102040816326531 X__b9,r3,band_LOW_PAIRED
        + 0.0102040816326531 X__b9,r3,band_UNPAIRED - GlobalBidder_W_ir__b9,r3
        + 0.0102040816326531 X__b9,r3,band_HIGH_PAIRED <= 0
 c2495: sv_function_helpervar_b6,r7__condX1
        - 0.158453086904266 sv_function_helpervar_b6,r7__Z1 <= 0
 c2496: - X__b2,r3,band_LOW_PAIRED
        + aux_cap_helper__b2,r3,band_LOW_PAIRED__condX1
        + aux_cap_helper__b2,r3,band_LOW_PAIRED__condX2  = 0
 c2497: X__b7,r4,band_LOW_PAIRED + X__b7,r4,band_UNPAIRED
        + X__b7,r4,band_HIGH_PAIRED - GlobalBidder_W_ir__b7,r4 >= 0
 c2498: - aux_cap_helper__b5,r0,band_LOW_PAIRED__condX2
        + aux_cap_helper__b5,r0,band_LOW_PAIRED__Z2 <= 0
 c2499: 2.8394335959 aux_cap_helper__b7,r8,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b7,r8,band_HIGH_PAIRED__condX1
        + 1.8253501688 aux_cap_helper__b7,r8,band_HIGH_PAIRED__condX3
        - 0.8112667416 aux_cap_helper__b7,r8,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b7,r8,band_HIGH_PAIRED__Z3
        - aux_cap_b7,r8,band_HIGH_PAIRED  = 0
 c2500: X__b9,r1,band_UNPAIRED + X__b9,r1,band_LOW_PAIRED
        + X__b9,r1,band_HIGH_PAIRED - GlobalBidder_W_ir__b9,r1 >= 0
 c2501: 2.8394335959 aux_cap_helper__b1,r5,band_HIGH_PAIRED__condX2
        + 1.8253501688 aux_cap_helper__b1,r5,band_HIGH_PAIRED__condX3
        + 2.0281668543 aux_cap_helper__b1,r5,band_HIGH_PAIRED__condX1
        + 1.21690011250031 aux_cap_helper__b1,r5,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b1,r5,band_HIGH_PAIRED__Z2
        - aux_cap_b1,r5,band_HIGH_PAIRED  = 0
 c2502: 0.158453086947678 sv_function_helpervar_b4,r7__Z2
        - sv_function_helpervar_b4,r7__condX2 <= 0
 c2503: sv_function_helpervar_b8,r8__Z3 + sv_function_helpervar_b8,r8__Z2
        + sv_function_helpervar_b8,r8__Z1  = 1
 c2504: - aux_cap_helper__b8,r2,band_HIGH_PAIRED__condX1 <= 0
 c2505: aux_cap_helper__b9,r6,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b9,r6,band_HIGH_PAIRED__Z2 <= 0
 c2506: sv_function_helpervar_b3,r7__Z2 + sv_function_helpervar_b3,r7__Z1
        + sv_function_helpervar_b3,r7__Z3  = 1
 c2507: 1.8253501688 aux_cap_helper__b7,r13,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b7,r13,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b7,r13,band_HIGH_PAIRED__condX1
        - 0.8112667416 aux_cap_helper__b7,r13,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b7,r13,band_HIGH_PAIRED__Z3
        - aux_cap_b7,r13,band_HIGH_PAIRED  = 0
 c2508: 3179.78098103941 aux_sv_b8,r4 - aux_Omega_b8,r4  = 0
 c2509: - 7.50122203625723 sv_function_helpervar_b8,r13__Z2
        + sv_function_helpervar_b8,r13__condX2 <= 0
 c2510: aux_cap_helper__b8,r3,band_LOW_PAIRED__Z2
        + aux_cap_helper__b8,r3,band_LOW_PAIRED__Z1  = 1
 c2511: - aux_cap_helper__b0,r4,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b0,r4,band_HIGH_PAIRED__Z3 <= 0
 c2512: - aux_cap_b4,r1,band_LOW_PAIRED
        + 4.4504289328 aux_cap_helper__b4,r1,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b4,r1,band_LOW_PAIRED__condX1
        - 1.2715511236 aux_cap_helper__b4,r1,band_LOW_PAIRED__Z2  = 0
 c2513: - aux_sv_b8,r11 + 31.8136631005 sv_function_helpervar_b8,r11__condX3
        + 1636.3575428388 sv_function_helpervar_b8,r11__condX1
        + 58.6126046563 sv_function_helpervar_b8,r11__condX2
        + 249.998555987508 sv_function_helpervar_b8,r11__Z2
        + 456.220851111678 sv_function_helpervar_b8,r11__Z3  = 0
 c2514: - aux_cap_helper__b4,r8,band_LOW_PAIRED__condX1 <= 0
 c2515: - aux_cap_helper__b6,r10,band_LOW_PAIRED__Z1
        + aux_cap_helper__b6,r10,band_LOW_PAIRED__condX1 <= 0
 c2516: aux_cap_helper__b7,r8,band_UNPAIRED__condX1
        - aux_cap_helper__b7,r8,band_UNPAIRED__Z1 <= 0
 c2517: - aux_cap_helper__b0,r12,band_HIGH_PAIRED__condX1 <= 0
 c2518: 372.500663204305 sv_function_helpervar_b3,r1__Z3
        + 174.567827851822 sv_function_helpervar_b3,r1__Z2
        + 19.1650426385 sv_function_helpervar_b3,r1__condX3
        + 1152.1877360178 sv_function_helpervar_b3,r1__condX1
        + 50.4873452333 sv_function_helpervar_b3,r1__condX2 - aux_sv_b3,r1  = 0
 c2519: - aux_cap_b5,r10,band_UNPAIRED
        - 0.253841888 aux_cap_helper__b5,r10,band_UNPAIRED__Z2
        + 0.888446608 aux_cap_helper__b5,r10,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b5,r10,band_UNPAIRED__condX1  = 0
 c2520: sv_function_helpervar_b4,r8__condX2
        - 5.84349771088166 sv_function_helpervar_b4,r8__Z2 <= 0
 c2521: 22.4846158588 sv_function_helpervar_b7,r3__condX3
        + 1255.6694400769 sv_function_helpervar_b7,r3__condX1
        + 49.5730297957 sv_function_helpervar_b7,r3__condX2
        + 191.109699307891 sv_function_helpervar_b7,r3__Z2
        + 380.630613215134 sv_function_helpervar_b7,r3__Z3 - aux_sv_b7,r3  = 0
 c2522: aux_cap_helper__b5,r0,band_UNPAIRED__Z1
        + aux_cap_helper__b5,r0,band_UNPAIRED__Z2  = 1
 c2523: - aux_cap_helper__b3,r2,band_LOW_PAIRED__condX1 <= 0
 c2524: 0.63460472 aux_cap_helper__b2,r13,band_UNPAIRED__condX1
        + 0.888446608 aux_cap_helper__b2,r13,band_UNPAIRED__condX2
        - 0.253841888 aux_cap_helper__b2,r13,band_UNPAIRED__Z2
        - aux_cap_b2,r13,band_UNPAIRED  = 0
 c2525: - aux_cap_helper__b3,r1,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b3,r1,band_HIGH_PAIRED__Z3 <= 0
 c2526: - 15.8453086887446 sv_function_helpervar_b1,r5__Z3
        + sv_function_helpervar_b1,r5__condX3 <= 0
 c2527: - aux_cap_helper__b8,r12,band_UNPAIRED__Z1
        + aux_cap_helper__b8,r12,band_UNPAIRED__condX1 <= 0
 c2528: - aux_cap_helper__b2,r11,band_UNPAIRED__condX1 <= 0
 c2529: sv_function_helpervar_b1,r6__condX1
        - 0.158453086995604 sv_function_helpervar_b1,r6__Z1 <= 0
 c2530: - 0.8112667416 aux_cap_helper__b3,r6,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b3,r6,band_HIGH_PAIRED__Z3
        - aux_cap_b3,r6,band_HIGH_PAIRED
        + 2.8394335959 aux_cap_helper__b3,r6,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b3,r6,band_HIGH_PAIRED__condX1
        + 1.8253501688 aux_cap_helper__b3,r6,band_HIGH_PAIRED__condX3  = 0
 c2531: aux_cap_helper__b6,r6,band_LOW_PAIRED__Z2
        - aux_cap_helper__b6,r6,band_LOW_PAIRED__condX2 <= 0
 c2532: - 2 aux_cap_helper__b6,r9,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b6,r9,band_HIGH_PAIRED__condX2 <= 0
 c2533: - 15.8453086887446 sv_function_helpervar_b4,r10__Z3
        + sv_function_helpervar_b4,r10__condX3 <= 0
 c2534: 2.0281668543 aux_cap_helper__b3,r5,band_HIGH_PAIRED__condX1
        + 2.8394335959 aux_cap_helper__b3,r5,band_HIGH_PAIRED__condX2
        + 1.8253501688 aux_cap_helper__b3,r5,band_HIGH_PAIRED__condX3
        - 0.8112667416 aux_cap_helper__b3,r5,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b3,r5,band_HIGH_PAIRED__Z3
        - aux_cap_b3,r5,band_HIGH_PAIRED  = 0
 c2535: - aux_cap_helper__b7,r0,band_UNPAIRED__condX1 <= 0
 c2536: - aux_cap_helper__b9,r11,band_LOW_PAIRED__condX2
        + aux_cap_helper__b9,r11,band_LOW_PAIRED__Z2 <= 0
 c2537: - X__b5,r11,band_HIGH_PAIRED
        + aux_cap_helper__b5,r11,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b5,r11,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b5,r11,band_HIGH_PAIRED__condX1  = 0
 c2538: - 3 aux_cap_helper__b3,r6,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b3,r6,band_HIGH_PAIRED__condX3 <= 0
 c2539: - sv_function_helpervar_b1,r4__condX1 <= 0
 c2540: 1.21690011250031 aux_cap_helper__b1,r7,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b1,r7,band_HIGH_PAIRED__Z2
        - aux_cap_b1,r7,band_HIGH_PAIRED
        + 1.8253501688 aux_cap_helper__b1,r7,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b1,r7,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b1,r7,band_HIGH_PAIRED__condX1  = 0
 c2541: 1.8253501688 aux_cap_helper__b7,r6,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b7,r6,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b7,r6,band_HIGH_PAIRED__condX1
        - 0.8112667416 aux_cap_helper__b7,r6,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b7,r6,band_HIGH_PAIRED__Z3
        - aux_cap_b7,r6,band_HIGH_PAIRED  = 0
 c2542: - X__b7,r13,band_HIGH_PAIRED
        + aux_cap_helper__b7,r13,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b7,r13,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b7,r13,band_HIGH_PAIRED__condX1  = 0
 c2543: - aux_Omega_b9,r8 + 330.794641751921 aux_sv_b9,r8  = 0
 c2544: - sv_function_helpervar_b3,r0__condX1 <= 0
 c2545: - sv_function_helpervar_b5,r1__condX3
        + 5.53380541572838 sv_function_helpervar_b5,r1__Z3 <= 0
 c2546: - X__b6,r0,band_LOW_PAIRED
        + aux_cap_helper__b6,r0,band_LOW_PAIRED__condX2
        + aux_cap_helper__b6,r0,band_LOW_PAIRED__condX1  = 0
 c2547: aux_cap_helper__b0,r12,band_UNPAIRED__Z2
        + aux_cap_helper__b0,r12,band_UNPAIRED__Z1  = 1
 c2548: sv_function_helpervar_b8,r12__condX1
        - 0.158453086927029 sv_function_helpervar_b8,r12__Z1 <= 0
 c2549: - 15.8453086887446 sv_function_helpervar_b3,r1__Z3
        + sv_function_helpervar_b3,r1__condX3 <= 0
 c2550: - aux_cap_helper__b6,r9,band_LOW_PAIRED__condX2
        + aux_cap_helper__b6,r9,band_LOW_PAIRED__Z2 <= 0
 c2551: sv_function_helpervar_b7,r11__Z1 + sv_function_helpervar_b7,r11__Z3
        + sv_function_helpervar_b7,r11__Z2  = 1
 c2552: - 2 aux_cap_helper__b7,r8,band_LOW_PAIRED__Z2
        + aux_cap_helper__b7,r8,band_LOW_PAIRED__condX2 <= 0
 c2553: - X__b9,r4,band_UNPAIRED + aux_cap_helper__b9,r4,band_UNPAIRED__condX1
        + aux_cap_helper__b9,r4,band_UNPAIRED__condX2  = 0
 c2554: - X__b1,r2,band_HIGH_PAIRED
        + aux_cap_helper__b1,r2,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b1,r2,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b1,r2,band_HIGH_PAIRED__condX2  = 0
 c2555: - 3 aux_cap_helper__b9,r0,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b9,r0,band_HIGH_PAIRED__condX3 <= 0
 c2556: sv_function_helpervar_b2,r8__condX1
        - 0.15845308692762 sv_function_helpervar_b2,r8__Z1 <= 0
 c2557: - aux_cap_helper__b8,r4,band_LOW_PAIRED__condX1 <= 0
 c2558: - aux_cap_helper__b5,r7,band_HIGH_PAIRED__condX1 <= 0
 c2559: - aux_cap_b9,r9,band_HIGH_PAIRED
        + 2.8394335959 aux_cap_helper__b9,r9,band_HIGH_PAIRED__condX2
        + 1.8253501688 aux_cap_helper__b9,r9,band_HIGH_PAIRED__condX3
        + 2.0281668543 aux_cap_helper__b9,r9,band_HIGH_PAIRED__condX1
        + 1.21690011250031 aux_cap_helper__b9,r9,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b9,r9,band_HIGH_PAIRED__Z2  = 0
 c2560: aux_cap_helper__b1,r6,band_UNPAIRED__Z1
        + aux_cap_helper__b1,r6,band_UNPAIRED__Z2  = 1
 c2561: aux_cap_helper__b6,r4,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b6,r4,band_HIGH_PAIRED__Z3 <= 0
 c2562: aux_cap_helper__b7,r8,band_LOW_PAIRED__Z1
        + aux_cap_helper__b7,r8,band_LOW_PAIRED__Z2  = 1
 c2563: - aux_cap_helper__b0,r3,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b0,r3,band_HIGH_PAIRED__Z3 <= 0
 c2564: aux_cap_helper__b0,r12,band_LOW_PAIRED__Z2
        + aux_cap_helper__b0,r12,band_LOW_PAIRED__Z1  = 1
 c2565: aux_cap_helper__b9,r6,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b9,r6,band_HIGH_PAIRED__Z1 <= 0
 c2566: aux_cap_b5,r13,band_LOW_PAIRED - aux_c_b5,r13
        + aux_cap_b5,r13,band_UNPAIRED + aux_cap_b5,r13,band_HIGH_PAIRED  = 0
 c2567: - sv_function_helpervar_b2,r0__condX2
        + 0.158453086896187 sv_function_helpervar_b2,r0__Z2 <= 0
 c2568: aux_cap_helper__b8,r8,band_UNPAIRED__condX1
        - aux_cap_helper__b8,r8,band_UNPAIRED__Z1 <= 0
 c2569: aux_cap_helper__b3,r0,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b3,r0,band_HIGH_PAIRED__condX2 <= 0
 c2570: - aux_cap_helper__b5,r11,band_LOW_PAIRED__Z1
        + aux_cap_helper__b5,r11,band_LOW_PAIRED__condX1 <= 0
 c2571: sv_function_helpervar_b2,r3__condX3
        - 15.8453086887446 sv_function_helpervar_b2,r3__Z3 <= 0
 c2572: - 0.158453086943648 sv_function_helpervar_b7,r8__Z1
        + sv_function_helpervar_b7,r8__condX1 <= 0
 c2573: aux_cap_b1,r9,band_UNPAIRED + aux_cap_b1,r9,band_LOW_PAIRED
        + aux_cap_b1,r9,band_HIGH_PAIRED - aux_c_b1,r9  = 0
 c2574: aux_cap_helper__b3,r3,band_UNPAIRED__Z1
        + aux_cap_helper__b3,r3,band_UNPAIRED__Z2  = 1
 c2575: - aux_Omega_b5,r3 + 298.411629179094 aux_sv_b5,r3  = 0
 c2576: - aux_cap_helper__b4,r1,band_UNPAIRED__condX1 <= 0
 c2577: aux_cap_helper__b8,r8,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b8,r8,band_UNPAIRED__Z2 <= 0
 c2578: - aux_cap_helper__b1,r12,band_UNPAIRED__condX2
        + aux_cap_helper__b1,r12,band_UNPAIRED__Z2 <= 0
 c2579: - aux_cap_b4,r8,band_LOW_PAIRED
        + 4.4504289328 aux_cap_helper__b4,r8,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b4,r8,band_LOW_PAIRED__condX1
        - 1.2715511236 aux_cap_helper__b4,r8,band_LOW_PAIRED__Z2  = 0
 c2580: aux_cap_helper__b2,r13,band_UNPAIRED__condX1
        + aux_cap_helper__b2,r13,band_UNPAIRED__condX2
        - X__b2,r13,band_UNPAIRED  = 0
 c2581: aux_cap_helper__b3,r9,band_LOW_PAIRED__Z1
        + aux_cap_helper__b3,r9,band_LOW_PAIRED__Z2  = 1
 c2582: 2 aux_cap_helper__b3,r0,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b3,r0,band_HIGH_PAIRED__condX3 <= 0
 c2583: - X__b3,r2,band_UNPAIRED + aux_cap_helper__b3,r2,band_UNPAIRED__condX2
        + aux_cap_helper__b3,r2,band_UNPAIRED__condX1  = 0
 c2584: aux_cap_helper__b4,r5,band_LOW_PAIRED__Z2
        + aux_cap_helper__b4,r5,band_LOW_PAIRED__Z1  = 1
 c2585: - X__b6,r11,band_LOW_PAIRED
        + aux_cap_helper__b6,r11,band_LOW_PAIRED__condX2
        + aux_cap_helper__b6,r11,band_LOW_PAIRED__condX1  = 0
 c2586: - 5.38866541022754 sv_function_helpervar_b3,r11__Z2
        + sv_function_helpervar_b3,r11__condX2 <= 0
 c2587: - aux_cap_helper__b2,r13,band_LOW_PAIRED__condX1 <= 0
 c2588: 1.21690011250031 aux_cap_helper__b5,r8,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b5,r8,band_HIGH_PAIRED__Z2
        + 1.8253501688 aux_cap_helper__b5,r8,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b5,r8,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b5,r8,band_HIGH_PAIRED__condX1
        - aux_cap_b5,r8,band_HIGH_PAIRED  = 0
 c2589: 302.757580971371 aux_sv_b8,r0 - aux_Omega_b8,r0  = 0
 c2590: 0.0102040816326531 X__b8,r10,band_HIGH_PAIRED
        + 0.0102040816326531 X__b8,r10,band_LOW_PAIRED
        - GlobalBidder_W_ir__b8,r10
        + 0.0102040816326531 X__b8,r10,band_UNPAIRED <= 0
 c2591: 4.4504289328 aux_cap_helper__b2,r7,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b2,r7,band_LOW_PAIRED__condX1
        - 1.2715511236 aux_cap_helper__b2,r7,band_LOW_PAIRED__Z2
        - aux_cap_b2,r7,band_LOW_PAIRED  = 0
 c2592: - aux_cap_helper__b0,r9,band_LOW_PAIRED__condX1 <= 0
 c2593: sv_function_helpervar_b4,r3__Z1 + sv_function_helpervar_b4,r3__Z2
        + sv_function_helpervar_b4,r3__Z3  = 1
 c2594: - aux_Omega_b6,r6 + 40.3758177550175 aux_sv_b6,r6  = 0
 c2595: sv_function_helpervar_b6,r2__condX1
        - 0.158453086936059 sv_function_helpervar_b6,r2__Z1 <= 0
 c2596: aux_cap_helper__b6,r12,band_UNPAIRED__Z2
        - aux_cap_helper__b6,r12,band_UNPAIRED__condX2 <= 0
 c2597: - aux_cap_helper__b0,r3,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b0,r3,band_HIGH_PAIRED__Z2 <= 0
 c2598: 0.0102040816326531 X__b7,r6,band_LOW_PAIRED
        + 0.0102040816326531 X__b7,r6,band_UNPAIRED - GlobalBidder_W_ir__b7,r6
        + 0.0102040816326531 X__b7,r6,band_HIGH_PAIRED <= 0
 c2599: aux_cap_helper__b0,r6,band_LOW_PAIRED__Z1
        + aux_cap_helper__b0,r6,band_LOW_PAIRED__Z2  = 1
 c2600: aux_cap_helper__b5,r4,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b5,r4,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b5,r4,band_HIGH_PAIRED__condX3
        - X__b5,r4,band_HIGH_PAIRED  = 0
 c2601: 0.63460472 aux_cap_helper__b5,r5,band_UNPAIRED__condX1
        + 0.888446608 aux_cap_helper__b5,r5,band_UNPAIRED__condX2
        - 0.253841888 aux_cap_helper__b5,r5,band_UNPAIRED__Z2
        - aux_cap_b5,r5,band_UNPAIRED  = 0
 c2602: - aux_cap_helper__b8,r1,band_HIGH_PAIRED__condX1 <= 0
 c2603: - aux_cap_helper__b2,r9,band_UNPAIRED__Z1
        + aux_cap_helper__b2,r9,band_UNPAIRED__condX1 <= 0
 c2604: 0.158453086923784 sv_function_helpervar_b3,r2__Z2
        - sv_function_helpervar_b3,r2__condX2 <= 0
 c2605: - sv_function_helpervar_b2,r2__condX1 <= 0
 c2606: 2.0281668543 aux_cap_helper__b7,r12,band_HIGH_PAIRED__condX1
        + 1.8253501688 aux_cap_helper__b7,r12,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b7,r12,band_HIGH_PAIRED__condX2
        + 1.21690011250031 aux_cap_helper__b7,r12,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b7,r12,band_HIGH_PAIRED__Z2
        - aux_cap_b7,r12,band_HIGH_PAIRED  = 0
 c2607: aux_cap_helper__b4,r11,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b4,r11,band_HIGH_PAIRED__Z3 <= 0
 c2608: - 0.253841888 aux_cap_helper__b0,r8,band_UNPAIRED__Z2
        + 0.888446608 aux_cap_helper__b0,r8,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b0,r8,band_UNPAIRED__condX1
        - aux_cap_b0,r8,band_UNPAIRED  = 0
 c2609: - 3 aux_cap_helper__b9,r10,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b9,r10,band_HIGH_PAIRED__condX3 <= 0
 c2610: aux_cap_b9,r8,band_UNPAIRED + aux_cap_b9,r8,band_LOW_PAIRED
        + aux_cap_b9,r8,band_HIGH_PAIRED - aux_c_b9,r8  = 0
 c2611: aux_cap_helper__b2,r0,band_LOW_PAIRED__condX1
        - aux_cap_helper__b2,r0,band_LOW_PAIRED__Z1 <= 0
 c2612: - aux_cap_helper__b3,r2,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b3,r2,band_HIGH_PAIRED__Z2 <= 0
 c2613: 0.158453086912404 sv_function_helpervar_b9,r12__Z2
        - sv_function_helpervar_b9,r12__condX2 <= 0
 c2614: aux_cap_helper__b4,r9,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b4,r9,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b4,r9,band_HIGH_PAIRED__condX1
        - X__b4,r9,band_HIGH_PAIRED  = 0
 c2615: aux_cap_helper__b2,r0,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b2,r0,band_LOW_PAIRED__Z2 <= 0
 c2616: aux_cap_helper__b8,r2,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b8,r2,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b8,r2,band_HIGH_PAIRED__Z3  = 1
 c2617: - aux_cap_helper__b3,r2,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b3,r2,band_HIGH_PAIRED__Z3 <= 0
 c2618: - X__b8,r0,band_HIGH_PAIRED
        + aux_cap_helper__b8,r0,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b8,r0,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b8,r0,band_HIGH_PAIRED__condX1  = 0
 c2619: - 6.29731842908851 sv_function_helpervar_b0,r8__Z2
        + sv_function_helpervar_b0,r8__condX2 <= 0
 c2620: - 0.8112667416 aux_cap_helper__b7,r11,band_HIGH_PAIRED__Z2
        + 1.8253501688 aux_cap_helper__b7,r11,band_HIGH_PAIRED__condX3
        + 2.0281668543 aux_cap_helper__b7,r11,band_HIGH_PAIRED__condX1
        + 2.8394335959 aux_cap_helper__b7,r11,band_HIGH_PAIRED__condX2
        - aux_cap_b7,r11,band_HIGH_PAIRED
        + 1.21690011250031 aux_cap_helper__b7,r11,band_HIGH_PAIRED__Z3  = 0
 c2621: - 0.158453086900387 sv_function_helpervar_b3,r11__Z1
        + sv_function_helpervar_b3,r11__condX1 <= 0
 c2622: - X__b1,r10,band_UNPAIRED
        + aux_cap_helper__b1,r10,band_UNPAIRED__condX2
        + aux_cap_helper__b1,r10,band_UNPAIRED__condX1  = 0
 c2623: aux_cap_helper__b6,r9,band_UNPAIRED__Z2
        + aux_cap_helper__b6,r9,band_UNPAIRED__Z1  = 1
 c2624: - 6.19049285847949 sv_function_helpervar_b9,r9__Z2
        + sv_function_helpervar_b9,r9__condX2 <= 0
 c2625: aux_cap_helper__b2,r4,band_UNPAIRED__Z2
        + aux_cap_helper__b2,r4,band_UNPAIRED__Z1  = 1
 c2626: GlobalBidder_PSI__i7,k1 - 20668026.8748422 GlobalBidder_hat{W}__i7,k1
        <= 0
 c2627: - sv_function_helpervar_b8,r2__condX1 <= 0
 c2628: 4.4504289328 aux_cap_helper__b0,r6,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b0,r6,band_LOW_PAIRED__condX1
        - 1.2715511236 aux_cap_helper__b0,r6,band_LOW_PAIRED__Z2
        - aux_cap_b0,r6,band_LOW_PAIRED  = 0
 c2629: 6.00310871588683 sv_function_helpervar_b6,r10__Z3
        - sv_function_helpervar_b6,r10__condX3 <= 0
 c2630: - X__b8,r0,band_UNPAIRED + aux_cap_helper__b8,r0,band_UNPAIRED__condX1
        + aux_cap_helper__b8,r0,band_UNPAIRED__condX2  = 0
 c2631: - 15.8453086887446 sv_function_helpervar_b1,r13__Z3
        + sv_function_helpervar_b1,r13__condX3 <= 0
 c2632: aux_cap_helper__b5,r8,band_UNPAIRED__condX1
        - aux_cap_helper__b5,r8,band_UNPAIRED__Z1 <= 0
 c2633: - aux_cap_helper__b8,r12,band_LOW_PAIRED__condX2
        + aux_cap_helper__b8,r12,band_LOW_PAIRED__Z2 <= 0
 c2634: - aux_cap_helper__b6,r13,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b6,r13,band_HIGH_PAIRED__Z2 <= 0
 c2635: aux_cap_helper__b1,r2,band_LOW_PAIRED__Z2
        + aux_cap_helper__b1,r2,band_LOW_PAIRED__Z1  = 1
 c2636: sv_function_helpervar_b5,r8__condX3
        - 15.8453086887446 sv_function_helpervar_b5,r8__Z3 <= 0
 c2637: aux_cap_helper__b5,r5,band_LOW_PAIRED__condX2
        + aux_cap_helper__b5,r5,band_LOW_PAIRED__condX1
        - X__b5,r5,band_LOW_PAIRED  = 0
 c2638: - aux_cap_helper__b0,r9,band_UNPAIRED__Z1
        + aux_cap_helper__b0,r9,band_UNPAIRED__condX1 <= 0
 c2639: aux_cap_helper__b8,r6,band_UNPAIRED__Z1
        + aux_cap_helper__b8,r6,band_UNPAIRED__Z2  = 1
 c2640: - aux_cap_helper__b3,r6,band_LOW_PAIRED__Z1
        + aux_cap_helper__b3,r6,band_LOW_PAIRED__condX1 <= 0
 c2641: - aux_cap_helper__b4,r5,band_LOW_PAIRED__condX1 <= 0
 c2642: aux_cap_b8,r6,band_LOW_PAIRED + aux_cap_b8,r6,band_UNPAIRED
        - aux_c_b8,r6 + aux_cap_b8,r6,band_HIGH_PAIRED  = 0
 c2643: sv_function_helpervar_b4,r10__Z2 + sv_function_helpervar_b4,r10__Z3
        + sv_function_helpervar_b4,r10__Z1  = 1
 c2644: - aux_cap_b9,r6,band_HIGH_PAIRED
        + 1.8253501688 aux_cap_helper__b9,r6,band_HIGH_PAIRED__condX3
        + 2.0281668543 aux_cap_helper__b9,r6,band_HIGH_PAIRED__condX1
        + 2.8394335959 aux_cap_helper__b9,r6,band_HIGH_PAIRED__condX2
        + 1.21690011250031 aux_cap_helper__b9,r6,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b9,r6,band_HIGH_PAIRED__Z2  = 0
 c2645: 0.158453086903266 sv_function_helpervar_b8,r13__Z2
        - sv_function_helpervar_b8,r13__condX2 <= 0
 c2646: aux_cap_helper__b4,r6,band_UNPAIRED__condX2
        + aux_cap_helper__b4,r6,band_UNPAIRED__condX1 - X__b4,r6,band_UNPAIRED
         = 0
 c2647: - aux_cap_helper__b6,r13,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b6,r13,band_HIGH_PAIRED__Z3 <= 0
 c2648: X__b2,r9,band_HIGH_PAIRED + X__b6,r9,band_HIGH_PAIRED
        + X__b5,r9,band_HIGH_PAIRED + X__b8,r9,band_HIGH_PAIRED
        + X__b9,r9,band_HIGH_PAIRED + X__b4,r9,band_HIGH_PAIRED
        + X__b0,r9,band_HIGH_PAIRED + X__b7,r9,band_HIGH_PAIRED
        + X__b3,r9,band_HIGH_PAIRED + X__b1,r9,band_HIGH_PAIRED <= 3
 c2649: - aux_cap_helper__b9,r0,band_UNPAIRED__condX1 <= 0
 c2650: 1.21690011250031 aux_cap_helper__b3,r3,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b3,r3,band_HIGH_PAIRED__Z2
        - aux_cap_b3,r3,band_HIGH_PAIRED
        + 1.8253501688 aux_cap_helper__b3,r3,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b3,r3,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b3,r3,band_HIGH_PAIRED__condX1  = 0
 c2651: aux_cap_helper__b7,r5,band_UNPAIRED__Z2
        - aux_cap_helper__b7,r5,band_UNPAIRED__condX2 <= 0
 c2652: - 2 aux_cap_helper__b0,r9,band_UNPAIRED__Z2
        + aux_cap_helper__b0,r9,band_UNPAIRED__condX2 <= 0
 c2653: - 2 aux_cap_helper__b3,r6,band_LOW_PAIRED__Z2
        + aux_cap_helper__b3,r6,band_LOW_PAIRED__condX2 <= 0
 c2654: - 1.2715511236 aux_cap_helper__b4,r10,band_LOW_PAIRED__Z2
        - aux_cap_b4,r10,band_LOW_PAIRED
        + 4.4504289328 aux_cap_helper__b4,r10,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b4,r10,band_LOW_PAIRED__condX1  = 0
 c2655: - aux_cap_helper__b0,r2,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b0,r2,band_HIGH_PAIRED__Z2 <= 0
 c2656: aux_cap_b1,r10,band_UNPAIRED + aux_cap_b1,r10,band_HIGH_PAIRED
        + aux_cap_b1,r10,band_LOW_PAIRED - aux_c_b1,r10  = 0
 c2657: aux_cap_helper__b3,r9,band_UNPAIRED__condX1
        - aux_cap_helper__b3,r9,band_UNPAIRED__Z1 <= 0
 c2658: - sv_function_helpervar_b8,r6__condX2
        + 0.158453086895541 sv_function_helpervar_b8,r6__Z2 <= 0
 c2659: aux_cap_helper__b8,r13,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b8,r13,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b8,r13,band_HIGH_PAIRED__Z3  = 1
 c2660: 5.79372069706504 sv_function_helpervar_b0,r0__Z3
        - sv_function_helpervar_b0,r0__condX3 <= 0
 c2661: - aux_sv_b8,r0 + 33.1161402269 sv_function_helpervar_b8,r0__condX3
        + 1636.3575440989 sv_function_helpervar_b8,r0__condX1
        + 56.2214075169 sv_function_helpervar_b8,r0__condX2
        + 435.582698983471 sv_function_helpervar_b8,r0__Z3
        + 250.377448562784 sv_function_helpervar_b8,r0__Z2  = 0
 c2662: X__b8,r5,band_LOW_PAIRED + X__b6,r5,band_LOW_PAIRED
        + X__b4,r5,band_LOW_PAIRED + X__b9,r5,band_LOW_PAIRED
        + X__b7,r5,band_LOW_PAIRED + X__b5,r5,band_LOW_PAIRED
        + X__b2,r5,band_LOW_PAIRED + X__b3,r5,band_LOW_PAIRED
        + X__b0,r5,band_LOW_PAIRED + X__b1,r5,band_LOW_PAIRED <= 2
 c2663: aux_cap_helper__b3,r5,band_LOW_PAIRED__condX2
        + aux_cap_helper__b3,r5,band_LOW_PAIRED__condX1
        - X__b3,r5,band_LOW_PAIRED  = 0
 c2664: - 3 aux_cap_helper__b4,r10,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b4,r10,band_HIGH_PAIRED__condX3 <= 0
 c2665: - 2 aux_cap_helper__b5,r11,band_LOW_PAIRED__Z2
        + aux_cap_helper__b5,r11,band_LOW_PAIRED__condX2 <= 0
 c2666: aux_cap_helper__b1,r2,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b1,r2,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b1,r2,band_HIGH_PAIRED__Z3  = 1
 c2667: aux_cap_b6,r5,band_HIGH_PAIRED - aux_c_b6,r5
        + aux_cap_b6,r5,band_UNPAIRED + aux_cap_b6,r5,band_LOW_PAIRED  = 0
 c2668: aux_cap_b6,r9,band_HIGH_PAIRED - aux_c_b6,r9
        + aux_cap_b6,r9,band_UNPAIRED + aux_cap_b6,r9,band_LOW_PAIRED  = 0
 c2669: - aux_cap_helper__b1,r7,band_LOW_PAIRED__condX2
        + aux_cap_helper__b1,r7,band_LOW_PAIRED__Z2 <= 0
 c2670: sv_function_helpervar_b1,r13__condX3
        + sv_function_helpervar_b1,r13__condX2
        + sv_function_helpervar_b1,r13__condX1 - aux_c_b1,r13  = 0
 c2671: aux_cap_helper__b3,r9,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b3,r9,band_UNPAIRED__Z2 <= 0
 c2672: aux_cap_helper__b6,r11,band_UNPAIRED__Z1
        + aux_cap_helper__b6,r11,band_UNPAIRED__Z2  = 1
 c2673: aux_cap_helper__b1,r7,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b1,r7,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b1,r7,band_HIGH_PAIRED__Z1  = 1
 c2674: - aux_cap_helper__b6,r0,band_UNPAIRED__condX1 <= 0
 c2675: aux_cap_helper__b9,r2,band_LOW_PAIRED__condX2
        + aux_cap_helper__b9,r2,band_LOW_PAIRED__condX1
        - X__b9,r2,band_LOW_PAIRED  = 0
 c2676: - aux_cap_helper__b0,r1,band_LOW_PAIRED__condX2
        + aux_cap_helper__b0,r1,band_LOW_PAIRED__Z2 <= 0
 c2677: - aux_cap_helper__b5,r13,band_UNPAIRED__condX1 <= 0
 c2678: 0.888446608 aux_cap_helper__b5,r9,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b5,r9,band_UNPAIRED__condX1
        - aux_cap_b5,r9,band_UNPAIRED
        - 0.253841888 aux_cap_helper__b5,r9,band_UNPAIRED__Z2  = 0
 c2679: aux_cap_helper__b2,r10,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b2,r10,band_HIGH_PAIRED__Z1 <= 0
 c2680: - aux_cap_b2,r7,band_UNPAIRED
        - 0.253841888 aux_cap_helper__b2,r7,band_UNPAIRED__Z2
        + 0.63460472 aux_cap_helper__b2,r7,band_UNPAIRED__condX1
        + 0.888446608 aux_cap_helper__b2,r7,band_UNPAIRED__condX2  = 0
 c2681: - aux_cap_helper__b5,r1,band_UNPAIRED__condX1 <= 0
 c2682: 7.12654647039803 sv_function_helpervar_b9,r8__Z3
        - sv_function_helpervar_b9,r8__condX3 <= 0
 c2683: - X__b0,r1,band_UNPAIRED + aux_cap_helper__b0,r1,band_UNPAIRED__condX1
        + aux_cap_helper__b0,r1,band_UNPAIRED__condX2  = 0
 c2684: - aux_cap_helper__b9,r8,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b9,r8,band_HIGH_PAIRED__condX1 <= 0
 c2685: aux_cap_helper__b0,r7,band_UNPAIRED__Z2
        - aux_cap_helper__b0,r7,band_UNPAIRED__condX2 <= 0
 c2686: aux_cap_helper__b2,r10,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b2,r10,band_HIGH_PAIRED__Z2 <= 0
 c2687: - aux_cap_helper__b2,r9,band_HIGH_PAIRED__condX1 <= 0
 c2688: aux_cap_helper__b7,r5,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b7,r5,band_LOW_PAIRED__Z2 <= 0
 c2689: 20668026.8748422 GlobalBidder_hat{W}__i8,k2 - GlobalBidder_PSI__i8,k2
        + aux_Omega_b8,r12 + aux_Omega_b8,r13 + aux_Omega_b8,r10
        + aux_Omega_b8,r11 + aux_Omega_b8,r5 + aux_Omega_b8,r6
        + aux_Omega_b8,r7 + aux_Omega_b8,r8 + aux_Omega_b8,r9 + aux_Omega_b8,r0
        + aux_Omega_b8,r1 + aux_Omega_b8,r2 + aux_Omega_b8,r3 + aux_Omega_b8,r4
        <= 20668026.8748422
 c2690: aux_cap_helper__b0,r8,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b0,r8,band_HIGH_PAIRED__Z3 <= 0
 c2691: - 2 aux_cap_helper__b9,r8,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b9,r8,band_HIGH_PAIRED__condX2 <= 0
 c2692: - 0.1584530869274 sv_function_helpervar_b7,r1__Z1
        + sv_function_helpervar_b7,r1__condX1 <= 0
 c2693: aux_cap_helper__b6,r3,band_LOW_PAIRED__Z2
        - aux_cap_helper__b6,r3,band_LOW_PAIRED__condX2 <= 0
 c2694: - sv_function_helpervar_b5,r7__condX1 <= 0
 c2695: aux_cap_helper__b7,r7,band_UNPAIRED__condX2
        + aux_cap_helper__b7,r7,band_UNPAIRED__condX1 - X__b7,r7,band_UNPAIRED
         = 0
 c2696: - aux_cap_helper__b2,r1,band_UNPAIRED__condX1 <= 0
 c2697: - aux_cap_helper__b8,r0,band_HIGH_PAIRED__condX1 <= 0
 c2698: - 6.59791083704279 sv_function_helpervar_b9,r1__Z2
        + sv_function_helpervar_b9,r1__condX2 <= 0
 c2699: aux_cap_helper__b7,r2,band_LOW_PAIRED__condX2
        + aux_cap_helper__b7,r2,band_LOW_PAIRED__condX1
        - X__b7,r2,band_LOW_PAIRED  = 0
 c2700: - 15.8453086887446 sv_function_helpervar_b8,r3__Z3
        + sv_function_helpervar_b8,r3__condX3 <= 0
 c2701: GlobalBidder_W_i_9 + 14 GlobalBidder_hat{W}__i9,k1 <= 27
 c2702: aux_cap_helper__b0,r11,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b0,r11,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b0,r11,band_HIGH_PAIRED__Z3  = 1
 c2703: - aux_cap_b5,r4,band_LOW_PAIRED
        - 1.2715511236 aux_cap_helper__b5,r4,band_LOW_PAIRED__Z2
        + 4.4504289328 aux_cap_helper__b5,r4,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b5,r4,band_LOW_PAIRED__condX1  = 0
 c2704: - aux_cap_helper__b0,r2,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b0,r2,band_HIGH_PAIRED__Z3 <= 0
 c2705: - sv_function_helpervar_b0,r6__condX1 <= 0
 c2706: - aux_cap_helper__b0,r10,band_HIGH_PAIRED__condX1 <= 0
 c2707: - sv_function_helpervar_b6,r13__condX3
        + 5.8935145383666 sv_function_helpervar_b6,r13__Z3 <= 0
 c2708: - aux_cap_helper__b9,r5,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b9,r5,band_HIGH_PAIRED__condX1 <= 0
 c2709: - aux_cap_helper__b0,r5,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b0,r5,band_HIGH_PAIRED__Z2 <= 0
 c2710: - 6.65803747744876 sv_function_helpervar_b7,r1__Z2
        + sv_function_helpervar_b7,r1__condX2 <= 0
 c2711: - X__b4,r10,band_LOW_PAIRED
        + aux_cap_helper__b4,r10,band_LOW_PAIRED__condX2
        + aux_cap_helper__b4,r10,band_LOW_PAIRED__condX1  = 0
 c2712: - 0.158453086891435 sv_function_helpervar_b6,r10__Z1
        + sv_function_helpervar_b6,r10__condX1 <= 0
 c2713: 373.466842191434 sv_function_helpervar_b3,r4__Z3
        + 174.528150598367 sv_function_helpervar_b3,r4__Z2
        + 1152.1877363888 sv_function_helpervar_b3,r4__condX1
        + 19.104066926 sv_function_helpervar_b3,r4__condX3
        + 50.7377490393 sv_function_helpervar_b3,r4__condX2 - aux_sv_b3,r4  = 0
 c2714: aux_cap_helper__b4,r10,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b4,r10,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b4,r10,band_HIGH_PAIRED__condX1
        - X__b4,r10,band_HIGH_PAIRED  = 0
 c2715: aux_cap_b3,r6,band_LOW_PAIRED + aux_cap_b3,r6,band_HIGH_PAIRED
        + aux_cap_b3,r6,band_UNPAIRED - aux_c_b3,r6  = 0
 c2716: 391.28947366523 sv_function_helpervar_b3,r6__Z3
        + 173.659377682706 sv_function_helpervar_b3,r6__Z2
        + 1152.1877366363 sv_function_helpervar_b3,r6__condX1
        + 56.220589004 sv_function_helpervar_b3,r6__condX2
        + 17.9792777645 sv_function_helpervar_b3,r6__condX3 - aux_sv_b3,r6  = 0
 c2717: - 2 aux_cap_helper__b6,r10,band_LOW_PAIRED__Z2
        + aux_cap_helper__b6,r10,band_LOW_PAIRED__condX2 <= 0
 c2718: aux_cap_helper__b5,r13,band_UNPAIRED__condX2
        + aux_cap_helper__b5,r13,band_UNPAIRED__condX1
        - X__b5,r13,band_UNPAIRED  = 0
 c2719: - aux_cap_helper__b8,r3,band_HIGH_PAIRED__condX1 <= 0
 c2720: - aux_cap_helper__b5,r6,band_UNPAIRED__condX2
        + aux_cap_helper__b5,r6,band_UNPAIRED__Z2 <= 0
 c2721: aux_cap_helper__b7,r5,band_LOW_PAIRED__condX1
        - aux_cap_helper__b7,r5,band_LOW_PAIRED__Z1 <= 0
 c2722: - 3 aux_cap_helper__b4,r13,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b4,r13,band_HIGH_PAIRED__condX3 <= 0
 c2723: aux_cap_helper__b9,r12,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b9,r12,band_HIGH_PAIRED__Z3 <= 0
 c2724: - X__b6,r7,band_HIGH_PAIRED
        + aux_cap_helper__b6,r7,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b6,r7,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b6,r7,band_HIGH_PAIRED__condX3  = 0
 c2725: - aux_cap_helper__b9,r13,band_UNPAIRED__condX2
        + aux_cap_helper__b9,r13,band_UNPAIRED__Z2 <= 0
 c2726: - sv_function_helpervar_b1,r9__condX3
        + 5.92566887890375 sv_function_helpervar_b1,r9__Z3 <= 0
 c2727: - 0.8112667416 aux_cap_helper__b2,r13,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b2,r13,band_HIGH_PAIRED__Z3
        - aux_cap_b2,r13,band_HIGH_PAIRED
        + 1.8253501688 aux_cap_helper__b2,r13,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b2,r13,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b2,r13,band_HIGH_PAIRED__condX1  = 0
 c2728: - aux_cap_b3,r8,band_HIGH_PAIRED
        + 1.21690011250031 aux_cap_helper__b3,r8,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b3,r8,band_HIGH_PAIRED__Z2
        + 1.8253501688 aux_cap_helper__b3,r8,band_HIGH_PAIRED__condX3
        + 2.0281668543 aux_cap_helper__b3,r8,band_HIGH_PAIRED__condX1
        + 2.8394335959 aux_cap_helper__b3,r8,band_HIGH_PAIRED__condX2  = 0
 c2729: - aux_cap_helper__b0,r13,band_HIGH_PAIRED__condX1 <= 0
 c2730: - X__b2,r7,band_UNPAIRED + aux_cap_helper__b2,r7,band_UNPAIRED__condX1
        + aux_cap_helper__b2,r7,band_UNPAIRED__condX2  = 0
 c2731: - aux_cap_b9,r6,band_LOW_PAIRED
        - 1.2715511236 aux_cap_helper__b9,r6,band_LOW_PAIRED__Z2
        + 3.1788778092 aux_cap_helper__b9,r6,band_LOW_PAIRED__condX1
        + 4.4504289328 aux_cap_helper__b9,r6,band_LOW_PAIRED__condX2  = 0
 c2732: aux_cap_helper__b6,r2,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b6,r2,band_HIGH_PAIRED__Z3 <= 0
 c2733: aux_cap_b9,r12,band_HIGH_PAIRED + aux_cap_b9,r12,band_UNPAIRED
        - aux_c_b9,r12 + aux_cap_b9,r12,band_LOW_PAIRED  = 0
 c2734: - aux_cap_helper__b0,r2,band_UNPAIRED__condX1 <= 0
 c2735: - aux_cap_helper__b5,r12,band_HIGH_PAIRED__condX1 <= 0
 c2736: - 3 aux_cap_helper__b3,r8,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b3,r8,band_HIGH_PAIRED__condX3 <= 0
 c2737: - 2 aux_cap_helper__b9,r7,band_UNPAIRED__Z2
        + aux_cap_helper__b9,r7,band_UNPAIRED__condX2 <= 0
 c2738: - aux_sv_b2,r8 + 114.7166939213 sv_function_helpervar_b2,r8__condX1
        + 5.5036558262 sv_function_helpervar_b2,r8__condX2
        + 1.8068902851 sv_function_helpervar_b2,r8__condX3
        + 17.3051430189124 sv_function_helpervar_b2,r8__Z2
        + 38.6922814959648 sv_function_helpervar_b2,r8__Z3  = 0
 c2739: - aux_Omega_b3,r13 + 13.8854406523933 aux_sv_b3,r13  = 0
 c2740: 173.144481942506 sv_function_helpervar_b3,r11__Z2
        + 399.525603353924 sv_function_helpervar_b3,r11__Z3
        + 1152.1877366294 sv_function_helpervar_b3,r11__condX1
        + 59.4701043667 sv_function_helpervar_b3,r11__condX2
        + 17.4594942847 sv_function_helpervar_b3,r11__condX3 - aux_sv_b3,r11
         = 0
 c2741: - aux_cap_b5,r5,band_HIGH_PAIRED
        + 1.21690011250031 aux_cap_helper__b5,r5,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b5,r5,band_HIGH_PAIRED__Z2
        + 2.0281668543 aux_cap_helper__b5,r5,band_HIGH_PAIRED__condX1
        + 1.8253501688 aux_cap_helper__b5,r5,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b5,r5,band_HIGH_PAIRED__condX2  = 0
 c2742: sv_function_helpervar_b4,r8__Z2 + sv_function_helpervar_b4,r8__Z1
        + sv_function_helpervar_b4,r8__Z3  = 1
 c2743: - aux_cap_helper__b6,r8,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b6,r8,band_HIGH_PAIRED__condX1 <= 0
 c2744: 3.1788778092 aux_cap_helper__b5,r10,band_LOW_PAIRED__condX1
        + 4.4504289328 aux_cap_helper__b5,r10,band_LOW_PAIRED__condX2
        - 1.2715511236 aux_cap_helper__b5,r10,band_LOW_PAIRED__Z2
        - aux_cap_b5,r10,band_LOW_PAIRED  = 0
 c2745: - aux_cap_helper__b7,r13,band_LOW_PAIRED__condX2
        + aux_cap_helper__b7,r13,band_LOW_PAIRED__Z2 <= 0
 c2746: - 0.8112667416 aux_cap_helper__b7,r4,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b7,r4,band_HIGH_PAIRED__Z3
        + 2.0281668543 aux_cap_helper__b7,r4,band_HIGH_PAIRED__condX1
        + 2.8394335959 aux_cap_helper__b7,r4,band_HIGH_PAIRED__condX2
        + 1.8253501688 aux_cap_helper__b7,r4,band_HIGH_PAIRED__condX3
        - aux_cap_b7,r4,band_HIGH_PAIRED  = 0
 c2747: - aux_cap_helper__b2,r6,band_UNPAIRED__condX2
        + aux_cap_helper__b2,r6,band_UNPAIRED__Z2 <= 0
 c2748: 3.1788778092 aux_cap_helper__b8,r3,band_LOW_PAIRED__condX1
        + 4.4504289328 aux_cap_helper__b8,r3,band_LOW_PAIRED__condX2
        - 1.2715511236 aux_cap_helper__b8,r3,band_LOW_PAIRED__Z2
        - aux_cap_b8,r3,band_LOW_PAIRED  = 0
 c2749: - 3 aux_cap_helper__b6,r5,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b6,r5,band_HIGH_PAIRED__condX3 <= 0
 c2750: - aux_cap_helper__b8,r7,band_LOW_PAIRED__condX1 <= 0
 c2751: - 2 aux_cap_helper__b6,r8,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b6,r8,band_HIGH_PAIRED__condX2 <= 0
 c2752: - X__b7,r13,band_LOW_PAIRED
        + aux_cap_helper__b7,r13,band_LOW_PAIRED__condX2
        + aux_cap_helper__b7,r13,band_LOW_PAIRED__condX1  = 0
 c2753: - aux_cap_b7,r0,band_UNPAIRED
        - 0.253841888 aux_cap_helper__b7,r0,band_UNPAIRED__Z2
        + 0.888446608 aux_cap_helper__b7,r0,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b7,r0,band_UNPAIRED__condX1  = 0
 c2754: - X__b2,r13,band_LOW_PAIRED
        + aux_cap_helper__b2,r13,band_LOW_PAIRED__condX2
        + aux_cap_helper__b2,r13,band_LOW_PAIRED__condX1  = 0
 c2755: sv_function_helpervar_b9,r10__Z3 + sv_function_helpervar_b9,r10__Z1
        + sv_function_helpervar_b9,r10__Z2  = 1
 c2756: aux_cap_helper__b3,r5,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b3,r5,band_HIGH_PAIRED__Z3 <= 0
 c2757: - aux_cap_b1,r9,band_HIGH_PAIRED
        + 1.8253501688 aux_cap_helper__b1,r9,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b1,r9,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b1,r9,band_HIGH_PAIRED__condX1
        - 0.8112667416 aux_cap_helper__b1,r9,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b1,r9,band_HIGH_PAIRED__Z3  = 0
 c2758: - aux_cap_helper__b7,r1,band_LOW_PAIRED__condX1 <= 0
 c2759: aux_cap_helper__b9,r5,band_UNPAIRED__Z2
        + aux_cap_helper__b9,r5,band_UNPAIRED__Z1  = 1
 c2760: - 0.253841888 aux_cap_helper__b8,r7,band_UNPAIRED__Z2
        - aux_cap_b8,r7,band_UNPAIRED
        + 0.63460472 aux_cap_helper__b8,r7,band_UNPAIRED__condX1
        + 0.888446608 aux_cap_helper__b8,r7,band_UNPAIRED__condX2  = 0
 c2761: - 3 aux_cap_helper__b9,r2,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b9,r2,band_HIGH_PAIRED__condX3 <= 0
 c2762: - 0.8112667416 aux_cap_helper__b7,r9,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b7,r9,band_HIGH_PAIRED__Z3
        - aux_cap_b7,r9,band_HIGH_PAIRED
        + 1.8253501688 aux_cap_helper__b7,r9,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b7,r9,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b7,r9,band_HIGH_PAIRED__condX1  = 0
 c2763: aux_cap_b0,r7,band_UNPAIRED + aux_cap_b0,r7,band_LOW_PAIRED
        + aux_cap_b0,r7,band_HIGH_PAIRED - aux_c_b0,r7  = 0
 c2764: - 0.8112667416 aux_cap_helper__b1,r4,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b1,r4,band_HIGH_PAIRED__Z3
        - aux_cap_b1,r4,band_HIGH_PAIRED
        + 2.8394335959 aux_cap_helper__b1,r4,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b1,r4,band_HIGH_PAIRED__condX1
        + 1.8253501688 aux_cap_helper__b1,r4,band_HIGH_PAIRED__condX3  = 0
 c2765: - 2 aux_cap_helper__b9,r5,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b9,r5,band_HIGH_PAIRED__condX2 <= 0
 c2766: - aux_cap_helper__b3,r10,band_UNPAIRED__condX2
        + aux_cap_helper__b3,r10,band_UNPAIRED__Z2 <= 0
 c2767: - aux_cap_helper__b9,r7,band_UNPAIRED__Z1
        + aux_cap_helper__b9,r7,band_UNPAIRED__condX1 <= 0
 c2768: - X__b1,r8,band_LOW_PAIRED
        + aux_cap_helper__b1,r8,band_LOW_PAIRED__condX1
        + aux_cap_helper__b1,r8,band_LOW_PAIRED__condX2  = 0
 c2769: sv_function_helpervar_b8,r4__condX2
        - 7.85656987996407 sv_function_helpervar_b8,r4__Z2 <= 0
 c2770: - aux_cap_helper__b5,r9,band_HIGH_PAIRED__condX1 <= 0
 c2771: 0.0102040816326531 X__b7,r12,band_HIGH_PAIRED
        + 0.0102040816326531 X__b7,r12,band_UNPAIRED
        + 0.0102040816326531 X__b7,r12,band_LOW_PAIRED
        - GlobalBidder_W_ir__b7,r12 <= 0
 c2772: - aux_cap_helper__b0,r5,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b0,r5,band_HIGH_PAIRED__Z3 <= 0
 c2773: - X__b0,r7,band_HIGH_PAIRED
        + aux_cap_helper__b0,r7,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b0,r7,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b0,r7,band_HIGH_PAIRED__condX1  = 0
 c2774: 0.63460472 aux_cap_helper__b3,r6,band_UNPAIRED__condX1
        + 0.888446608 aux_cap_helper__b3,r6,band_UNPAIRED__condX2
        - 0.253841888 aux_cap_helper__b3,r6,band_UNPAIRED__Z2
        - aux_cap_b3,r6,band_UNPAIRED  = 0
 c2775: aux_cap_helper__b4,r3,band_LOW_PAIRED__condX1
        + aux_cap_helper__b4,r3,band_LOW_PAIRED__condX2
        - X__b4,r3,band_LOW_PAIRED  = 0
 c2776: 2.8394335959 aux_cap_helper__b3,r1,band_HIGH_PAIRED__condX2
        + 1.8253501688 aux_cap_helper__b3,r1,band_HIGH_PAIRED__condX3
        + 2.0281668543 aux_cap_helper__b3,r1,band_HIGH_PAIRED__condX1
        + 1.21690011250031 aux_cap_helper__b3,r1,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b3,r1,band_HIGH_PAIRED__Z2
        - aux_cap_b3,r1,band_HIGH_PAIRED  = 0
 c2777: 0.63460472 aux_cap_helper__b6,r8,band_UNPAIRED__condX1
        + 0.888446608 aux_cap_helper__b6,r8,band_UNPAIRED__condX2
        - aux_cap_b6,r8,band_UNPAIRED
        - 0.253841888 aux_cap_helper__b6,r8,band_UNPAIRED__Z2  = 0
 c2778: 2 aux_cap_helper__b0,r1,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b0,r1,band_HIGH_PAIRED__condX3 <= 0
 c2779: - aux_cap_helper__b6,r0,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b6,r0,band_HIGH_PAIRED__Z3 <= 0
 c2780: aux_cap_helper__b2,r0,band_UNPAIRED__condX2
        + aux_cap_helper__b2,r0,band_UNPAIRED__condX1 - X__b2,r0,band_UNPAIRED
         = 0
 c2781: sv_function_helpervar_b7,r5__condX2
        - 8.07903836345579 sv_function_helpervar_b7,r5__Z2 <= 0
 c2782: - 1.2715511236 aux_cap_helper__b2,r12,band_LOW_PAIRED__Z2
        + 4.4504289328 aux_cap_helper__b2,r12,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b2,r12,band_LOW_PAIRED__condX1
        - aux_cap_b2,r12,band_LOW_PAIRED  = 0
 c2783: - aux_Omega_b6,r13 + 21.5051180945533 aux_sv_b6,r13  = 0
 c2784: aux_cap_helper__b6,r6,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b6,r6,band_HIGH_PAIRED__Z1 <= 0
 c2785: 17.3646117757086 sv_function_helpervar_b2,r6__Z2
        + 37.470286780847 sv_function_helpervar_b2,r6__Z3 - aux_sv_b2,r6
        + 5.128347539 sv_function_helpervar_b2,r6__condX2
        + 114.7166939405 sv_function_helpervar_b2,r6__condX1
        + 1.8840105697 sv_function_helpervar_b2,r6__condX3  = 0
 c2786: aux_cap_helper__b9,r10,band_LOW_PAIRED__condX1
        - aux_cap_helper__b9,r10,band_LOW_PAIRED__Z1 <= 0
 c2787: - 20668026.8748422 GlobalBidder_hat{W}__i8,k1 + GlobalBidder_PSI__i8,k1
        <= 0
 c2788: - aux_cap_helper__b1,r2,band_UNPAIRED__condX1 <= 0
 c2789: 0.158453086894239 sv_function_helpervar_b5,r13__Z2
        - sv_function_helpervar_b5,r13__condX2 <= 0
 c2790: aux_cap_helper__b6,r6,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b6,r6,band_HIGH_PAIRED__Z2 <= 0
 c2791: - v__9 + 0.90202 GlobalBidder_PSI__i9,k2
        + 0.97551 GlobalBidder_PSI__i9,k1 + GlobalBidder_PSI__i9,k0  = 0
 c2792: 942.4308566455 sv_function_helpervar_b6,r7__condX1
        + 43.2558051899 sv_function_helpervar_b6,r7__condX2
        + 15.2297898708 sv_function_helpervar_b6,r7__condX3
        + 311.757346490437 sv_function_helpervar_b6,r7__Z3
        + 142.477062570442 sv_function_helpervar_b6,r7__Z2 - aux_sv_b6,r7  = 0
 c2793: aux_cap_helper__b0,r7,band_LOW_PAIRED__Z1
        + aux_cap_helper__b0,r7,band_LOW_PAIRED__Z2  = 1
 c2794: 0.158453086934085 sv_function_helpervar_b1,r10__Z2
        - sv_function_helpervar_b1,r10__condX2 <= 0
 c2795: sv_function_helpervar_b8,r10__condX3
        + sv_function_helpervar_b8,r10__condX2
        + sv_function_helpervar_b8,r10__condX1 - aux_c_b8,r10  = 0
 c2796: 14 GlobalBidder_hat{W}__i8,k0 + GlobalBidder_W_i_8 <= 28
 c2797: aux_cap_helper__b9,r10,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b9,r10,band_LOW_PAIRED__Z2 <= 0
 c2798: X__b5,r4,band_UNPAIRED + X__b4,r4,band_UNPAIRED
        + X__b7,r4,band_UNPAIRED + X__b8,r4,band_UNPAIRED
        + X__b9,r4,band_UNPAIRED + X__b3,r4,band_UNPAIRED
        + X__b0,r4,band_UNPAIRED + X__b2,r4,band_UNPAIRED
        + X__b1,r4,band_UNPAIRED + X__b6,r4,band_UNPAIRED <= 2
 c2799: aux_cap_helper__b0,r1,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b0,r1,band_HIGH_PAIRED__condX2 <= 0
 c2800: - aux_cap_helper__b2,r12,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b2,r12,band_HIGH_PAIRED__condX1 <= 0
 c2801: aux_cap_helper__b7,r9,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b7,r9,band_HIGH_PAIRED__condX2 <= 0
 c2802: - aux_cap_helper__b1,r6,band_UNPAIRED__condX2
        + aux_cap_helper__b1,r6,band_UNPAIRED__Z2 <= 0
 c2803: - aux_Omega_b9,r3 + 925.145665760936 aux_sv_b9,r3  = 0
 c2804: - sv_function_helpervar_b8,r7__condX2
        + 0.158453086983862 sv_function_helpervar_b8,r7__Z2 <= 0
 c2805: sv_function_helpervar_b6,r7__condX1
        + sv_function_helpervar_b6,r7__condX2
        + sv_function_helpervar_b6,r7__condX3 - aux_c_b6,r7  = 0
 c2806: sv_function_helpervar_b3,r2__Z1 + sv_function_helpervar_b3,r2__Z3
        + sv_function_helpervar_b3,r2__Z2  = 1
 c2807: - aux_cap_helper__b0,r7,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b0,r7,band_HIGH_PAIRED__Z3 <= 0
 c2808: - aux_cap_helper__b2,r0,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b2,r0,band_HIGH_PAIRED__condX1 <= 0
 c2809: - X__b4,r6,band_HIGH_PAIRED
        + aux_cap_helper__b4,r6,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b4,r6,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b4,r6,band_HIGH_PAIRED__condX2  = 0
 c2810: - sv_function_helpervar_b8,r11__condX1 <= 0
 c2811: - 2 aux_cap_helper__b2,r12,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b2,r12,band_HIGH_PAIRED__condX2 <= 0
 c2812: - aux_cap_b3,r10,band_LOW_PAIRED
        + 4.4504289328 aux_cap_helper__b3,r10,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b3,r10,band_LOW_PAIRED__condX1
        - 1.2715511236 aux_cap_helper__b3,r10,band_LOW_PAIRED__Z2  = 0
 c2813: - aux_cap_helper__b9,r7,band_LOW_PAIRED__condX1 <= 0
 c2814: sv_function_helpervar_b1,r2__condX1 - aux_c_b1,r2
        + sv_function_helpervar_b1,r2__condX3
        + sv_function_helpervar_b1,r2__condX2  = 0
 c2815: - 3 aux_cap_helper__b3,r4,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b3,r4,band_HIGH_PAIRED__condX3 <= 0
 c2816: aux_cap_helper__b4,r8,band_UNPAIRED__condX1
        - aux_cap_helper__b4,r8,band_UNPAIRED__Z1 <= 0
 c2817: 3.1788778092 aux_cap_helper__b6,r9,band_LOW_PAIRED__condX1
        + 4.4504289328 aux_cap_helper__b6,r9,band_LOW_PAIRED__condX2
        - aux_cap_b6,r9,band_LOW_PAIRED
        - 1.2715511236 aux_cap_helper__b6,r9,band_LOW_PAIRED__Z2  = 0
 c2818: - aux_cap_helper__b9,r2,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b9,r2,band_HIGH_PAIRED__condX1 <= 0
 c2819: 0.0102040816326531 X__b9,r7,band_LOW_PAIRED - GlobalBidder_W_ir__b9,r7
        + 0.0102040816326531 X__b9,r7,band_HIGH_PAIRED
        + 0.0102040816326531 X__b9,r7,band_UNPAIRED <= 0
 c2820: 652.087439924221 sv_function_helpervar_b9,r9__Z3
        + 302.010865671521 sv_function_helpervar_b9,r9__Z2
        + 1995.292526449 sv_function_helpervar_b9,r9__condX1
        + 89.2970581519 sv_function_helpervar_b9,r9__condX2
        + 32.7463791905 sv_function_helpervar_b9,r9__condX3 - aux_sv_b9,r9  = 0
 c2821: sv_function_helpervar_b1,r10__condX1
        + sv_function_helpervar_b1,r10__condX2
        + sv_function_helpervar_b1,r10__condX3 - aux_c_b1,r10  = 0
 c2822: aux_cap_helper__b0,r5,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b0,r5,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b0,r5,band_HIGH_PAIRED__Z3  = 1
 c2823: - 2 aux_cap_helper__b2,r0,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b2,r0,band_HIGH_PAIRED__condX2 <= 0
 c2824: aux_cap_helper__b9,r12,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b9,r12,band_HIGH_PAIRED__Z1 <= 0
 c2825: - aux_cap_helper__b3,r3,band_LOW_PAIRED__Z1
        + aux_cap_helper__b3,r3,band_LOW_PAIRED__condX1 <= 0
 c2826: aux_cap_helper__b9,r3,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b9,r3,band_HIGH_PAIRED__Z3 <= 0
 c2827: aux_cap_helper__b4,r8,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b4,r8,band_UNPAIRED__Z2 <= 0
 c2828: aux_cap_helper__b5,r10,band_LOW_PAIRED__Z1
        + aux_cap_helper__b5,r10,band_LOW_PAIRED__Z2  = 1
 c2829: 0.158453086943132 sv_function_helpervar_b1,r3__Z2
        - sv_function_helpervar_b1,r3__condX2 <= 0
 c2830: 0.158453086913542 sv_function_helpervar_b4,r12__Z2
        - sv_function_helpervar_b4,r12__condX2 <= 0
 c2831: - sv_function_helpervar_b5,r0__condX2
        + 0.158453086925932 sv_function_helpervar_b5,r0__Z2 <= 0
 c2832: aux_cap_helper__b9,r12,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b9,r12,band_HIGH_PAIRED__Z2 <= 0
 c2833: aux_cap_helper__b2,r6,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b2,r6,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b2,r6,band_HIGH_PAIRED__Z1  = 1
 c2834: - 2 aux_cap_helper__b3,r3,band_LOW_PAIRED__Z2
        + aux_cap_helper__b3,r3,band_LOW_PAIRED__condX2 <= 0
 c2835: - 14 GlobalBidder_hat{W}__i8,k0 + GlobalBidder_W_i_8 >= 0
 c2836: aux_cap_helper__b0,r11,band_UNPAIRED__Z1
        + aux_cap_helper__b0,r11,band_UNPAIRED__Z2  = 1
 c2837: 0.888446608 aux_cap_helper__b3,r13,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b3,r13,band_UNPAIRED__condX1
        - 0.253841888 aux_cap_helper__b3,r13,band_UNPAIRED__Z2
        - aux_cap_b3,r13,band_UNPAIRED  = 0
 c2838: sv_function_helpervar_b6,r3__condX2
        - 6.01419287738594 sv_function_helpervar_b6,r3__Z2 <= 0
 c2839: sv_function_helpervar_b8,r3__Z1 + sv_function_helpervar_b8,r3__Z3
        + sv_function_helpervar_b8,r3__Z2  = 1
 c2840: 2.8394335959 aux_cap_helper__b9,r3,band_HIGH_PAIRED__condX2
        + 1.8253501688 aux_cap_helper__b9,r3,band_HIGH_PAIRED__condX3
        + 2.0281668543 aux_cap_helper__b9,r3,band_HIGH_PAIRED__condX1
        - aux_cap_b9,r3,band_HIGH_PAIRED
        - 0.8112667416 aux_cap_helper__b9,r3,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b9,r3,band_HIGH_PAIRED__Z3  = 0
 c2841: - X__b7,r5,band_UNPAIRED + aux_cap_helper__b7,r5,band_UNPAIRED__condX1
        + aux_cap_helper__b7,r5,band_UNPAIRED__condX2  = 0
 c2842: 0.888446608 aux_cap_helper__b1,r0,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b1,r0,band_UNPAIRED__condX1
        - aux_cap_b1,r0,band_UNPAIRED
        - 0.253841888 aux_cap_helper__b1,r0,band_UNPAIRED__Z2  = 0
 c2843: aux_cap_helper__b3,r12,band_UNPAIRED__condX1
        - aux_cap_helper__b3,r12,band_UNPAIRED__Z1 <= 0
 c2844: aux_cap_helper__b9,r11,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b9,r11,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b9,r11,band_HIGH_PAIRED__Z3  = 1
 c2845: - aux_cap_helper__b0,r7,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b0,r7,band_HIGH_PAIRED__Z2 <= 0
 c2846: aux_cap_helper__b3,r13,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b3,r13,band_LOW_PAIRED__Z2 <= 0
 c2847: sv_function_helpervar_b5,r12__Z2 + sv_function_helpervar_b5,r12__Z1
        + sv_function_helpervar_b5,r12__Z3  = 1
 c2848: aux_cap_helper__b0,r10,band_UNPAIRED__condX1
        - aux_cap_helper__b0,r10,band_UNPAIRED__Z1 <= 0
 c2849: - aux_cap_helper__b8,r5,band_HIGH_PAIRED__condX1 <= 0
 c2850: 6.22043635189836 sv_function_helpervar_b4,r11__Z3
        - sv_function_helpervar_b4,r11__condX3 <= 0
 c2851: - aux_cap_helper__b6,r11,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b6,r11,band_HIGH_PAIRED__Z2 <= 0
 c2852: 0.888446608 aux_cap_helper__b1,r13,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b1,r13,band_UNPAIRED__condX1
        - 0.253841888 aux_cap_helper__b1,r13,band_UNPAIRED__Z2
        - aux_cap_b1,r13,band_UNPAIRED  = 0
 c2853: 6.23552468667465 sv_function_helpervar_b1,r5__Z3
        - sv_function_helpervar_b1,r5__condX3 <= 0
 c2854: aux_cap_helper__b1,r5,band_UNPAIRED__Z1
        + aux_cap_helper__b1,r5,band_UNPAIRED__Z2  = 1
 c2855: 1.21690011250031 aux_cap_helper__b1,r0,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b1,r0,band_HIGH_PAIRED__Z2
        + 1.8253501688 aux_cap_helper__b1,r0,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b1,r0,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b1,r0,band_HIGH_PAIRED__condX1
        - aux_cap_b1,r0,band_HIGH_PAIRED  = 0
 c2856: - sv_function_helpervar_b7,r12__condX2
        + 0.158453086918105 sv_function_helpervar_b7,r12__Z2 <= 0
 c2857: sv_function_helpervar_b8,r12__condX3
        - 15.8453086887446 sv_function_helpervar_b8,r12__Z3 <= 0
 c2858: - sv_function_helpervar_b0,r7__condX1 <= 0
 c2859: - aux_cap_helper__b2,r2,band_UNPAIRED__condX1 <= 0
 c2860: - 2 aux_cap_helper__b6,r7,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b6,r7,band_HIGH_PAIRED__condX2 <= 0
 c2861: - 2 aux_cap_helper__b6,r9,band_UNPAIRED__Z2
        + aux_cap_helper__b6,r9,band_UNPAIRED__condX2 <= 0
 c2862: aux_cap_helper__b8,r11,band_LOW_PAIRED__condX1
        - aux_cap_helper__b8,r11,band_LOW_PAIRED__Z1 <= 0
 c2863: - aux_cap_helper__b6,r11,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b6,r11,band_HIGH_PAIRED__Z3 <= 0
 c2864: sv_function_helpervar_b7,r5__condX3
        - 15.8453086887446 sv_function_helpervar_b7,r5__Z3 <= 0
 c2865: aux_cap_helper__b8,r4,band_LOW_PAIRED__condX1
        + aux_cap_helper__b8,r4,band_LOW_PAIRED__condX2
        - X__b8,r4,band_LOW_PAIRED  = 0
 c2866: - aux_c_b5,r2 + sv_function_helpervar_b5,r2__condX2
        + sv_function_helpervar_b5,r2__condX3
        + sv_function_helpervar_b5,r2__condX1  = 0
 c2867: - 0.158453086908613 sv_function_helpervar_b3,r12__Z1
        + sv_function_helpervar_b3,r12__condX1 <= 0
 c2868: - aux_cap_helper__b7,r11,band_UNPAIRED__condX1 <= 0
 c2869: - aux_cap_b5,r3,band_HIGH_PAIRED
        + 1.21690011250031 aux_cap_helper__b5,r3,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b5,r3,band_HIGH_PAIRED__Z2
        + 2.0281668543 aux_cap_helper__b5,r3,band_HIGH_PAIRED__condX1
        + 1.8253501688 aux_cap_helper__b5,r3,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b5,r3,band_HIGH_PAIRED__condX2  = 0
 c2870: aux_cap_helper__b3,r13,band_UNPAIRED__condX1
        - aux_cap_helper__b3,r13,band_UNPAIRED__Z1 <= 0
 c2871: aux_cap_helper__b8,r11,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b8,r11,band_LOW_PAIRED__Z2 <= 0
 c2872: 784.297743342326 aux_sv_b4,r12 - aux_Omega_b4,r12  = 0
 c2873: aux_cap_helper__b0,r0,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b0,r0,band_HIGH_PAIRED__condX2 <= 0
 c2874: - aux_Omega_b3,r8 + 191.305836620374 aux_sv_b3,r8  = 0
 c2875: - 0.158453086892038 sv_function_helpervar_b6,r5__Z1
        + sv_function_helpervar_b6,r5__condX1 <= 0
 c2876: aux_cap_helper__b6,r10,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b6,r10,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b6,r10,band_HIGH_PAIRED__Z2  = 1
 c2877: - aux_cap_helper__b0,r6,band_UNPAIRED__condX2
        + aux_cap_helper__b0,r6,band_UNPAIRED__Z2 <= 0
 c2878: 68.8563260254 sv_function_helpervar_b9,r6__condX2
        + 40.2027724856 sv_function_helpervar_b9,r6__condX3
        + 1995.2925268226 sv_function_helpervar_b9,r6__condX1
        + 305.249762778592 sv_function_helpervar_b9,r6__Z2
        + 533.938586458852 sv_function_helpervar_b9,r6__Z3 - aux_sv_b9,r6  = 0
 c2879: - X__b7,r5,band_HIGH_PAIRED
        + aux_cap_helper__b7,r5,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b7,r5,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b7,r5,band_HIGH_PAIRED__condX1  = 0
 c2880: - aux_cap_helper__b4,r0,band_UNPAIRED__condX1 <= 0
 c2881: aux_cap_helper__b0,r13,band_LOW_PAIRED__Z1
        + aux_cap_helper__b0,r13,band_LOW_PAIRED__Z2  = 1
 c2882: - 3 aux_cap_helper__b3,r3,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b3,r3,band_HIGH_PAIRED__condX3 <= 0
 c2883: aux_cap_helper__b7,r12,band_LOW_PAIRED__Z2
        + aux_cap_helper__b7,r12,band_LOW_PAIRED__Z1  = 1
 c2884: - sv_function_helpervar_b1,r13__condX1 <= 0
 c2885: 191.678134867159 sv_function_helpervar_b7,r4__Z2
        + 357.775600158105 sv_function_helpervar_b7,r4__Z3
        + 1255.6694398067 sv_function_helpervar_b7,r4__condX1
        + 45.9856237568 sv_function_helpervar_b7,r4__condX2
        + 23.9269994378 sv_function_helpervar_b7,r4__condX3 - aux_sv_b7,r4  = 0
 c2886: aux_cap_helper__b3,r12,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b3,r12,band_UNPAIRED__Z2 <= 0
 c2887: aux_cap_helper__b4,r1,band_UNPAIRED__Z2
        + aux_cap_helper__b4,r1,band_UNPAIRED__Z1  = 1
 c2888: - sv_function_helpervar_b6,r5__condX1 <= 0
 c2889: - 0.158453086936884 sv_function_helpervar_b4,r1__Z1
        + sv_function_helpervar_b4,r1__condX1 <= 0
 c2890: - aux_cap_helper__b3,r8,band_LOW_PAIRED__condX1 <= 0
 c2891: - 2 aux_cap_helper__b4,r12,band_LOW_PAIRED__Z2
        + aux_cap_helper__b4,r12,band_LOW_PAIRED__condX2 <= 0
 c2892: 2 aux_cap_helper__b7,r9,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b7,r9,band_HIGH_PAIRED__condX3 <= 0
 c2893: sv_function_helpervar_b7,r5__condX1
        - 0.158453086897541 sv_function_helpervar_b7,r5__Z1 <= 0
 c2894: 25.1881763983 sv_function_helpervar_b7,r11__condX3
        + 43.5270764182 sv_function_helpervar_b7,r11__condX2
        + 1255.6694396954 sv_function_helpervar_b7,r11__condX1 - aux_sv_b7,r11
        + 337.79186190784 sv_function_helpervar_b7,r11__Z3
        + 192.067699281717 sv_function_helpervar_b7,r11__Z2  = 0
 c2895: - aux_Omega_b0,r13 + 33.3248107173472 aux_sv_b0,r13  = 0
 c2896: aux_cap_helper__b0,r8,band_UNPAIRED__Z2
        + aux_cap_helper__b0,r8,band_UNPAIRED__Z1  = 1
 c2897: - aux_cap_helper__b0,r6,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b0,r6,band_HIGH_PAIRED__Z2 <= 0
 c2898: - 2 aux_cap_helper__b8,r8,band_LOW_PAIRED__Z2
        + aux_cap_helper__b8,r8,band_LOW_PAIRED__condX2 <= 0
 c2899: - 6.24539851566527 sv_function_helpervar_b4,r7__Z2
        + sv_function_helpervar_b4,r7__condX2 <= 0
 c2900: - aux_cap_helper__b2,r11,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b2,r11,band_HIGH_PAIRED__condX1 <= 0
 c2901: aux_cap_helper__b2,r6,band_LOW_PAIRED__condX1
        - aux_cap_helper__b2,r6,band_LOW_PAIRED__Z1 <= 0
 c2902: 18.7838749456 sv_function_helpervar_b3,r8__condX3
        + 52.1227952315 sv_function_helpervar_b3,r8__condX2
        + 1152.1877362465 sv_function_helpervar_b3,r8__condX1
        + 378.540382960722 sv_function_helpervar_b3,r8__Z3
        + 174.308685752641 sv_function_helpervar_b3,r8__Z2 - aux_sv_b3,r8  = 0
 c2903: - aux_cap_b1,r1,band_HIGH_PAIRED
        - 0.8112667416 aux_cap_helper__b1,r1,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b1,r1,band_HIGH_PAIRED__Z3
        + 2.0281668543 aux_cap_helper__b1,r1,band_HIGH_PAIRED__condX1
        + 1.8253501688 aux_cap_helper__b1,r1,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b1,r1,band_HIGH_PAIRED__condX2  = 0
 c2904: 190.700795848101 sv_function_helpervar_b7,r1__Z2
        + 393.751398933842 sv_function_helpervar_b7,r1__Z3
        + 52.1536261499 sv_function_helpervar_b7,r1__condX2
        + 1255.6694398587 sv_function_helpervar_b7,r1__condX1
        + 21.6565609451 sv_function_helpervar_b7,r1__condX3 - aux_sv_b7,r1  = 0
 c2905: - 1.2715511236 aux_cap_helper__b3,r5,band_LOW_PAIRED__Z2
        - aux_cap_b3,r5,band_LOW_PAIRED
        + 4.4504289328 aux_cap_helper__b3,r5,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b3,r5,band_LOW_PAIRED__condX1  = 0
 c2906: - 3 aux_cap_helper__b9,r13,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b9,r13,band_HIGH_PAIRED__condX3 <= 0
 c2907: sv_function_helpervar_b1,r6__condX3
        - 15.8453086887446 sv_function_helpervar_b1,r6__Z3 <= 0
 c2908: aux_cap_helper__b5,r13,band_UNPAIRED__Z1
        + aux_cap_helper__b5,r13,band_UNPAIRED__Z2  = 1
 c2909: - sv_function_helpervar_b6,r2__condX3
        + 6.03603404051893 sv_function_helpervar_b6,r2__Z3 <= 0
 c2910: sv_function_helpervar_b0,r13__Z2 + sv_function_helpervar_b0,r13__Z1
        + sv_function_helpervar_b0,r13__Z3  = 1
 c2911: aux_cap_helper__b3,r13,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b3,r13,band_UNPAIRED__Z2 <= 0
 c2912: - aux_cap_helper__b5,r4,band_HIGH_PAIRED__condX1 <= 0
 c2913: - X__b2,r11,band_HIGH_PAIRED
        + aux_cap_helper__b2,r11,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b2,r11,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b2,r11,band_HIGH_PAIRED__condX1  = 0
 c2914: aux_cap_helper__b2,r6,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b2,r6,band_LOW_PAIRED__Z2 <= 0
 c2915: 2 aux_cap_helper__b0,r0,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b0,r0,band_HIGH_PAIRED__condX3 <= 0
 c2916: - 2 aux_cap_helper__b2,r11,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b2,r11,band_HIGH_PAIRED__condX2 <= 0
 c2917: - sv_function_helpervar_b1,r5__condX1 <= 0
 c2918: aux_cap_helper__b7,r10,band_LOW_PAIRED__condX2
        + aux_cap_helper__b7,r10,band_LOW_PAIRED__condX1
        - X__b7,r10,band_LOW_PAIRED  = 0
 c2919: - X__b9,r9,band_LOW_PAIRED
        + aux_cap_helper__b9,r9,band_LOW_PAIRED__condX1
        + aux_cap_helper__b9,r9,band_LOW_PAIRED__condX2  = 0
 c2920: - X__b2,r9,band_UNPAIRED + aux_cap_helper__b2,r9,band_UNPAIRED__condX2
        + aux_cap_helper__b2,r9,band_UNPAIRED__condX1  = 0
 c2921: - 6.31923003595998 sv_function_helpervar_b3,r1__Z2
        + sv_function_helpervar_b3,r1__condX2 <= 0
 c2922: aux_cap_helper__b7,r5,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b7,r5,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b7,r5,band_HIGH_PAIRED__Z2  = 1
 c2923: - aux_cap_helper__b7,r10,band_LOW_PAIRED__condX2
        + aux_cap_helper__b7,r10,band_LOW_PAIRED__Z2 <= 0
 c2924: sv_function_helpervar_b7,r1__Z1 + sv_function_helpervar_b7,r1__Z2
        + sv_function_helpervar_b7,r1__Z3  = 1
 c2925: - aux_cap_helper__b7,r6,band_UNPAIRED__condX2
        + aux_cap_helper__b7,r6,band_UNPAIRED__Z2 <= 0
 c2926: - sv_function_helpervar_b3,r3__condX3
        + 5.74780324936675 sv_function_helpervar_b3,r3__Z3 <= 0
 c2927: - X__b4,r13,band_HIGH_PAIRED
        + aux_cap_helper__b4,r13,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b4,r13,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b4,r13,band_HIGH_PAIRED__condX2  = 0
 c2928: - sv_function_helpervar_b7,r11__condX3
        + 7.94617793150088 sv_function_helpervar_b7,r11__Z3 <= 0
 c2929: 1.21690011250031 aux_cap_helper__b2,r11,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b2,r11,band_HIGH_PAIRED__Z2
        - aux_cap_b2,r11,band_HIGH_PAIRED
        + 1.8253501688 aux_cap_helper__b2,r11,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b2,r11,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b2,r11,band_HIGH_PAIRED__condX1  = 0
 c2930: - aux_cap_helper__b6,r0,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b6,r0,band_HIGH_PAIRED__Z2 <= 0
 c2931: aux_cap_helper__b7,r12,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b7,r12,band_LOW_PAIRED__Z2 <= 0
 c2932: aux_cap_b4,r8,band_LOW_PAIRED + aux_cap_b4,r8,band_UNPAIRED
        - aux_c_b4,r8 + aux_cap_b4,r8,band_HIGH_PAIRED  = 0
 c2933: - sv_function_helpervar_b7,r11__condX2
        + 0.158453086948001 sv_function_helpervar_b7,r11__Z2 <= 0
 c2934: sv_function_helpervar_b3,r13__condX1
        - 0.158453086929357 sv_function_helpervar_b3,r13__Z1 <= 0
 c2935: sv_function_helpervar_b5,r8__condX1
        + sv_function_helpervar_b5,r8__condX3
        + sv_function_helpervar_b5,r8__condX2 - aux_c_b5,r8  = 0
 c2936: - 0.8112667416 aux_cap_helper__b7,r1,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b7,r1,band_HIGH_PAIRED__Z3
        - aux_cap_b7,r1,band_HIGH_PAIRED
        + 2.8394335959 aux_cap_helper__b7,r1,band_HIGH_PAIRED__condX2
        + 1.8253501688 aux_cap_helper__b7,r1,band_HIGH_PAIRED__condX3
        + 2.0281668543 aux_cap_helper__b7,r1,band_HIGH_PAIRED__condX1  = 0
 c2937: - aux_cap_helper__b8,r8,band_LOW_PAIRED__Z1
        + aux_cap_helper__b8,r8,band_LOW_PAIRED__condX1 <= 0
 c2938: - aux_cap_helper__b9,r13,band_LOW_PAIRED__Z1
        + aux_cap_helper__b9,r13,band_LOW_PAIRED__condX1 <= 0
 c2939: aux_cap_helper__b4,r7,band_UNPAIRED__Z2
        - aux_cap_helper__b4,r7,band_UNPAIRED__condX2 <= 0
 c2940: - 0.15845308698947 sv_function_helpervar_b1,r1__Z1
        + sv_function_helpervar_b1,r1__condX1 <= 0
 c2941: aux_cap_helper__b9,r11,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b9,r11,band_HIGH_PAIRED__Z2 <= 0
 c2942: - aux_cap_b1,r12,band_LOW_PAIRED
        - 1.2715511236 aux_cap_helper__b1,r12,band_LOW_PAIRED__Z2
        + 3.1788778092 aux_cap_helper__b1,r12,band_LOW_PAIRED__condX1
        + 4.4504289328 aux_cap_helper__b1,r12,band_LOW_PAIRED__condX2  = 0
 c2943: aux_cap_helper__b5,r9,band_LOW_PAIRED__Z2
        - aux_cap_helper__b5,r9,band_LOW_PAIRED__condX2 <= 0
 c2944: 0.158453087021984 sv_function_helpervar_b8,r1__Z2
        - sv_function_helpervar_b8,r1__condX2 <= 0
 c2945: - aux_c_b8,r4 + sv_function_helpervar_b8,r4__condX3
        + sv_function_helpervar_b8,r4__condX1
        + sv_function_helpervar_b8,r4__condX2  = 0
 c2946: - X__b5,r8,band_UNPAIRED + aux_cap_helper__b5,r8,band_UNPAIRED__condX2
        + aux_cap_helper__b5,r8,band_UNPAIRED__condX1  = 0
 c2947: - aux_cap_helper__b2,r5,band_HIGH_PAIRED__condX1 <= 0
 c2948: aux_cap_helper__b1,r1,band_LOW_PAIRED__Z1
        + aux_cap_helper__b1,r1,band_LOW_PAIRED__Z2  = 1
 c2949: - aux_cap_helper__b8,r4,band_HIGH_PAIRED__condX1 <= 0
 c2950: sv_function_helpervar_b1,r4__condX3
        - 15.8453086887446 sv_function_helpervar_b1,r4__Z3 <= 0
 c2951: aux_cap_helper__b1,r1,band_LOW_PAIRED__Z2
        - aux_cap_helper__b1,r1,band_LOW_PAIRED__condX2 <= 0
 c2952: - 2 aux_cap_helper__b9,r13,band_LOW_PAIRED__Z2
        + aux_cap_helper__b9,r13,band_LOW_PAIRED__condX2 <= 0
 c2953: - aux_cap_helper__b0,r13,band_UNPAIRED__condX1 <= 0
 c2954: aux_cap_helper__b9,r4,band_UNPAIRED__Z2
        - aux_cap_helper__b9,r4,band_UNPAIRED__condX2 <= 0
 c2955: GlobalBidder_W_i_7 + 14 GlobalBidder_hat{W}__i7,k2 <= 26
 c2956: - 0.158453086934238 sv_function_helpervar_b0,r1__Z1
        + sv_function_helpervar_b0,r1__condX1 <= 0
 c2957: - 2 aux_cap_helper__b8,r5,band_LOW_PAIRED__Z2
        + aux_cap_helper__b8,r5,band_LOW_PAIRED__condX2 <= 0
 c2958: aux_cap_helper__b7,r12,band_LOW_PAIRED__condX1
        - aux_cap_helper__b7,r12,band_LOW_PAIRED__Z1 <= 0
 c2959: sv_function_helpervar_b8,r11__condX2
        - 7.69516567267345 sv_function_helpervar_b8,r11__Z2 <= 0
 c2960: aux_cap_helper__b9,r1,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b9,r1,band_HIGH_PAIRED__Z1 <= 0
 c2961: 4.4504289328 aux_cap_helper__b2,r0,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b2,r0,band_LOW_PAIRED__condX1
        - aux_cap_b2,r0,band_LOW_PAIRED
        - 1.2715511236 aux_cap_helper__b2,r0,band_LOW_PAIRED__Z2  = 0
 c2962: aux_cap_helper__b9,r4,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b9,r4,band_HIGH_PAIRED__Z3 <= 0
 c2963: - sv_function_helpervar_b3,r10__condX2
        + 0.158453086896816 sv_function_helpervar_b3,r10__Z2 <= 0
 c2964: aux_cap_helper__b4,r5,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b4,r5,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b4,r5,band_HIGH_PAIRED__Z2  = 1
 c2965: sv_function_helpervar_b9,r5__condX2
        - 7.77426771339398 sv_function_helpervar_b9,r5__Z2 <= 0
 c2966: aux_cap_helper__b9,r7,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b9,r7,band_HIGH_PAIRED__Z2 <= 0
 c2967: - aux_cap_helper__b4,r12,band_LOW_PAIRED__Z1
        + aux_cap_helper__b4,r12,band_LOW_PAIRED__condX1 <= 0
 c2968: aux_cap_b4,r1,band_UNPAIRED + aux_cap_b4,r1,band_HIGH_PAIRED
        + aux_cap_b4,r1,band_LOW_PAIRED - aux_c_b4,r1  = 0
 c2969: - aux_cap_helper__b5,r10,band_UNPAIRED__Z1
        + aux_cap_helper__b5,r10,band_UNPAIRED__condX1 <= 0
 c2970: - 0.158453087015465 sv_function_helpervar_b9,r2__Z1
        + sv_function_helpervar_b9,r2__condX1 <= 0
 c2971: - sv_function_helpervar_b7,r4__condX1 <= 0
 c2972: - 0.8112667416 aux_cap_helper__b5,r9,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b5,r9,band_HIGH_PAIRED__Z3
        - aux_cap_b5,r9,band_HIGH_PAIRED
        + 1.8253501688 aux_cap_helper__b5,r9,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b5,r9,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b5,r9,band_HIGH_PAIRED__condX1  = 0
 c2973: - sv_function_helpervar_b9,r4__condX2
        + 0.158453086901837 sv_function_helpervar_b9,r4__Z2 <= 0
 c2974: sv_function_helpervar_b6,r6__condX3
        - 15.8453086887446 sv_function_helpervar_b6,r6__Z3 <= 0
 c2975: sv_function_helpervar_b7,r3__condX2
        - 6.99638282066709 sv_function_helpervar_b7,r3__Z2 <= 0
 c2976: - aux_cap_helper__b6,r13,band_LOW_PAIRED__Z1
        + aux_cap_helper__b6,r13,band_LOW_PAIRED__condX1 <= 0
 c2977: - sv_function_helpervar_b8,r8__condX2
        + 0.158453086894935 sv_function_helpervar_b8,r8__Z2 <= 0
 c2978: aux_cap_b5,r3,band_HIGH_PAIRED - aux_c_b5,r3
        + aux_cap_b5,r3,band_LOW_PAIRED + aux_cap_b5,r3,band_UNPAIRED  = 0
 c2979: - aux_cap_helper__b1,r9,band_LOW_PAIRED__condX1 <= 0
 c2980: - aux_cap_helper__b6,r1,band_LOW_PAIRED__condX1 <= 0
 c2981: aux_cap_helper__b1,r3,band_UNPAIRED__condX2
        + aux_cap_helper__b1,r3,band_UNPAIRED__condX1 - X__b1,r3,band_UNPAIRED
         = 0
 c2982: - 2 aux_cap_helper__b5,r10,band_UNPAIRED__Z2
        + aux_cap_helper__b5,r10,band_UNPAIRED__condX2 <= 0
 c2983: - aux_cap_helper__b7,r8,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b7,r8,band_HIGH_PAIRED__Z2 <= 0
 c2984: - aux_cap_helper__b6,r6,band_UNPAIRED__condX2
        + aux_cap_helper__b6,r6,band_UNPAIRED__Z2 <= 0
 c2985: 1.21690011250031 aux_cap_helper__b2,r12,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b2,r12,band_HIGH_PAIRED__Z2
        + 2.0281668543 aux_cap_helper__b2,r12,band_HIGH_PAIRED__condX1
        + 2.8394335959 aux_cap_helper__b2,r12,band_HIGH_PAIRED__condX2
        + 1.8253501688 aux_cap_helper__b2,r12,band_HIGH_PAIRED__condX3
        - aux_cap_b2,r12,band_HIGH_PAIRED  = 0
 c2986: aux_cap_b6,r1,band_LOW_PAIRED + aux_cap_b6,r1,band_HIGH_PAIRED
        - aux_c_b6,r1 + aux_cap_b6,r1,band_UNPAIRED  = 0
 c2987: aux_cap_helper__b6,r1,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b6,r1,band_HIGH_PAIRED__condX2 <= 0
 c2988: 1.21690011250031 aux_cap_helper__b7,r0,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b7,r0,band_HIGH_PAIRED__Z2
        - aux_cap_b7,r0,band_HIGH_PAIRED
        + 1.8253501688 aux_cap_helper__b7,r0,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b7,r0,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b7,r0,band_HIGH_PAIRED__condX1  = 0
 c2989: - 2 aux_cap_helper__b6,r13,band_LOW_PAIRED__Z2
        + aux_cap_helper__b6,r13,band_LOW_PAIRED__condX2 <= 0
 c2990: aux_cap_helper__b7,r7,band_LOW_PAIRED__Z1
        + aux_cap_helper__b7,r7,band_LOW_PAIRED__Z2  = 1
 c2991: - aux_cap_helper__b3,r5,band_LOW_PAIRED__condX1 <= 0
 c2992: - aux_cap_helper__b0,r6,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b0,r6,band_HIGH_PAIRED__Z3 <= 0
 c2993: - aux_cap_helper__b7,r8,band_LOW_PAIRED__Z1
        + aux_cap_helper__b7,r8,band_LOW_PAIRED__condX1 <= 0
 c2994: - aux_cap_helper__b7,r8,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b7,r8,band_HIGH_PAIRED__Z3 <= 0
 c2995: - aux_sv_b2,r0 + 114.7166939441 sv_function_helpervar_b2,r0__condX1
        + 5.39349566 sv_function_helpervar_b2,r0__condX2
        + 1.8277712554 sv_function_helpervar_b2,r0__condX3
        + 17.3225982374796 sv_function_helpervar_b2,r0__Z2
        + 38.3614160762359 sv_function_helpervar_b2,r0__Z3  = 0
 c2996: - X__b7,r9,band_LOW_PAIRED
        + aux_cap_helper__b7,r9,band_LOW_PAIRED__condX2
        + aux_cap_helper__b7,r9,band_LOW_PAIRED__condX1  = 0
 c2997: - 3 aux_cap_helper__b3,r9,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b3,r9,band_HIGH_PAIRED__condX3 <= 0
 c2998: aux_cap_helper__b9,r7,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b9,r7,band_HIGH_PAIRED__Z1 <= 0
 c2999: - aux_cap_helper__b5,r11,band_HIGH_PAIRED__condX1 <= 0
 c3000: X__b9,r10,band_LOW_PAIRED + X__b9,r10,band_UNPAIRED
        - GlobalBidder_W_ir__b9,r10 + X__b9,r10,band_HIGH_PAIRED >= 0
 c3001: aux_cap_helper__b3,r0,band_LOW_PAIRED__condX1
        - aux_cap_helper__b3,r0,band_LOW_PAIRED__Z1 <= 0
 c3002: 2 aux_cap_helper__b6,r1,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b6,r1,band_HIGH_PAIRED__condX3 <= 0
 c3003: sv_function_helpervar_b0,r3__Z3 + sv_function_helpervar_b0,r3__Z2
        + sv_function_helpervar_b0,r3__Z1  = 1
 c3004: 1.21690011250031 aux_cap_helper__b5,r4,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b5,r4,band_HIGH_PAIRED__Z2
        + 2.0281668543 aux_cap_helper__b5,r4,band_HIGH_PAIRED__condX1
        + 2.8394335959 aux_cap_helper__b5,r4,band_HIGH_PAIRED__condX2
        + 1.8253501688 aux_cap_helper__b5,r4,band_HIGH_PAIRED__condX3
        - aux_cap_b5,r4,band_HIGH_PAIRED  = 0
 c3005: - aux_sv_b1,r11 + 150.9476061546 sv_function_helpervar_b1,r11__condX1
        + 6.7993071152 sv_function_helpervar_b1,r11__condX2
        + 2.4673902702 sv_function_helpervar_b1,r11__condX3
        + 49.4890474955682 sv_function_helpervar_b1,r11__Z3
        + 22.8407429538745 sv_function_helpervar_b1,r11__Z2  = 0
 c3006: - 1.2715511236 aux_cap_helper__b3,r3,band_LOW_PAIRED__Z2
        - aux_cap_b3,r3,band_LOW_PAIRED
        + 3.1788778092 aux_cap_helper__b3,r3,band_LOW_PAIRED__condX1
        + 4.4504289328 aux_cap_helper__b3,r3,band_LOW_PAIRED__condX2  = 0
 c3007: - aux_cap_helper__b6,r7,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b6,r7,band_HIGH_PAIRED__condX1 <= 0
 c3008: 24.8161600553401 sv_function_helpervar_b0,r10__Z2
        + 52.7505436779559 sv_function_helpervar_b0,r10__Z3
        + 163.7425743967 sv_function_helpervar_b0,r10__condX1
        + 2.7354444301 sv_function_helpervar_b0,r10__condX3
        + 7.1273860074 sv_function_helpervar_b0,r10__condX2 - aux_sv_b0,r10
         = 0
 c3009: - 3 aux_cap_helper__b0,r9,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b0,r9,band_HIGH_PAIRED__condX3 <= 0
 c3010: aux_cap_helper__b4,r4,band_LOW_PAIRED__Z1
        + aux_cap_helper__b4,r4,band_LOW_PAIRED__Z2  = 1
 c3011: 0.158453087005849 sv_function_helpervar_b7,r9__Z2
        - sv_function_helpervar_b7,r9__condX2 <= 0
 c3012: aux_cap_helper__b9,r8,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b9,r8,band_UNPAIRED__Z2 <= 0
 c3013: - aux_cap_b7,r5,band_HIGH_PAIRED
        + 1.21690011250031 aux_cap_helper__b7,r5,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b7,r5,band_HIGH_PAIRED__Z2
        + 1.8253501688 aux_cap_helper__b7,r5,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b7,r5,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b7,r5,band_HIGH_PAIRED__condX1  = 0
 c3014: sv_function_helpervar_b3,r0__condX1
        - 0.158453086901643 sv_function_helpervar_b3,r0__Z1 <= 0
 c3015: - aux_cap_helper__b5,r5,band_HIGH_PAIRED__condX1 <= 0
 c3016: aux_cap_helper__b9,r1,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b9,r1,band_HIGH_PAIRED__Z2 <= 0
 c3017: - aux_c_b3,r13 + aux_cap_b3,r13,band_LOW_PAIRED
        + aux_cap_b3,r13,band_HIGH_PAIRED + aux_cap_b3,r13,band_UNPAIRED  = 0
 c3018: sv_function_helpervar_b6,r11__condX3
        - 15.8453086887446 sv_function_helpervar_b6,r11__Z3 <= 0
 c3019: - aux_c_b3,r5 + sv_function_helpervar_b3,r5__condX1
        + sv_function_helpervar_b3,r5__condX2
        + sv_function_helpervar_b3,r5__condX3  = 0
 c3020: - sv_function_helpervar_b4,r9__condX3
        + 5.95597226282117 sv_function_helpervar_b4,r9__Z3 <= 0
 c3021: sv_function_helpervar_b9,r0__condX3
        - 15.8453086887446 sv_function_helpervar_b9,r0__Z3 <= 0
 c3022: aux_cap_helper__b7,r7,band_UNPAIRED__Z1
        + aux_cap_helper__b7,r7,band_UNPAIRED__Z2  = 1
 c3023: aux_cap_helper__b0,r4,band_LOW_PAIRED__Z2
        - aux_cap_helper__b0,r4,band_LOW_PAIRED__condX2 <= 0
 c3024: - aux_cap_helper__b1,r9,band_UNPAIRED__Z1
        + aux_cap_helper__b1,r9,band_UNPAIRED__condX1 <= 0
 c3025: aux_cap_helper__b3,r13,band_LOW_PAIRED__condX1
        - aux_cap_helper__b3,r13,band_LOW_PAIRED__Z1 <= 0
 c3026: - aux_cap_helper__b3,r1,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b3,r1,band_HIGH_PAIRED__Z2 <= 0
 c3027: aux_cap_helper__b9,r8,band_UNPAIRED__condX1
        - aux_cap_helper__b9,r8,band_UNPAIRED__Z1 <= 0
 c3028: - aux_c_b6,r11 + aux_cap_b6,r11,band_HIGH_PAIRED
        + aux_cap_b6,r11,band_LOW_PAIRED + aux_cap_b6,r11,band_UNPAIRED  = 0
 c3029: - 2 aux_cap_helper__b1,r9,band_UNPAIRED__Z2
        + aux_cap_helper__b1,r9,band_UNPAIRED__condX2 <= 0
 c3030: 293.053256800003 sv_function_helpervar_b4,r6__Z3
        + 133.513855331108 sv_function_helpervar_b4,r6__Z2
        + 14.2239839681 sv_function_helpervar_b4,r6__condX3
        + 883.4028070269 sv_function_helpervar_b4,r6__condX1
        + 40.794702399 sv_function_helpervar_b4,r6__condX2 - aux_sv_b4,r6  = 0
 c3031: X__b7,r8,band_UNPAIRED - GlobalBidder_W_ir__b7,r8
        + X__b7,r8,band_HIGH_PAIRED + X__b7,r8,band_LOW_PAIRED >= 0
 c3032: aux_cap_helper__b3,r9,band_UNPAIRED__condX2
        + aux_cap_helper__b3,r9,band_UNPAIRED__condX1 - X__b3,r9,band_UNPAIRED
         = 0
 c3033: 15.8192254884 sv_function_helpervar_b5,r9__condX3
        + 44.9294729567 sv_function_helpervar_b5,r9__condX2
        + 978.9018726667 sv_function_helpervar_b5,r9__condX1 - aux_sv_b5,r9
        + 147.990809874421 sv_function_helpervar_b5,r9__Z2
        + 323.821057661236 sv_function_helpervar_b5,r9__Z3  = 0
 c3034: sv_function_helpervar_b1,r8__condX2
        + sv_function_helpervar_b1,r8__condX3
        + sv_function_helpervar_b1,r8__condX1 - aux_c_b1,r8  = 0
 c3035: aux_cap_helper__b9,r2,band_LOW_PAIRED__Z2
        - aux_cap_helper__b9,r2,band_LOW_PAIRED__condX2 <= 0
 c3036: - X__b5,r9,band_LOW_PAIRED
        + aux_cap_helper__b5,r9,band_LOW_PAIRED__condX2
        + aux_cap_helper__b5,r9,band_LOW_PAIRED__condX1  = 0
 c3037: sv_function_helpervar_b1,r12__condX2
        - 5.6159506028775 sv_function_helpervar_b1,r12__Z2 <= 0
 c3038: aux_cap_helper__b5,r1,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b5,r1,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b5,r1,band_HIGH_PAIRED__condX3
        - X__b5,r1,band_HIGH_PAIRED  = 0
 c3039: - sv_function_helpervar_b1,r3__condX1 <= 0
 c3040: aux_cap_helper__b1,r11,band_UNPAIRED__Z2
        + aux_cap_helper__b1,r11,band_UNPAIRED__Z1  = 1
 c3041: aux_cap_helper__b3,r0,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b3,r0,band_LOW_PAIRED__Z2 <= 0
 c3042: aux_cap_helper__b3,r8,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b3,r8,band_UNPAIRED__Z2 <= 0
 c3043: aux_cap_helper__b5,r4,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b5,r4,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b5,r4,band_HIGH_PAIRED__Z1  = 1
 c3044: - sv_function_helpervar_b6,r10__condX1 <= 0
 c3045: aux_cap_helper__b8,r10,band_UNPAIRED__Z2
        - aux_cap_helper__b8,r10,band_UNPAIRED__condX2 <= 0
 c3046: - aux_cap_helper__b5,r0,band_UNPAIRED__condX1 <= 0
 c3047: GlobalBidder_W_i_9 + 13 GlobalBidder_hat{W}__i9,k2 >= 13
 c3048: 1.21690011250031 aux_cap_helper__b9,r2,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b9,r2,band_HIGH_PAIRED__Z2
        - aux_cap_b9,r2,band_HIGH_PAIRED
        + 2.8394335959 aux_cap_helper__b9,r2,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b9,r2,band_HIGH_PAIRED__condX1
        + 1.8253501688 aux_cap_helper__b9,r2,band_HIGH_PAIRED__condX3  = 0
 c3049: - aux_cap_b1,r7,band_UNPAIRED
        + 0.63460472 aux_cap_helper__b1,r7,band_UNPAIRED__condX1
        + 0.888446608 aux_cap_helper__b1,r7,band_UNPAIRED__condX2
        - 0.253841888 aux_cap_helper__b1,r7,band_UNPAIRED__Z2  = 0
 c3050: sv_function_helpervar_b5,r10__condX1
        + sv_function_helpervar_b5,r10__condX2
        + sv_function_helpervar_b5,r10__condX3 - aux_c_b5,r10  = 0
 c3051: aux_cap_helper__b8,r12,band_LOW_PAIRED__condX2
        + aux_cap_helper__b8,r12,band_LOW_PAIRED__condX1
        - X__b8,r12,band_LOW_PAIRED  = 0
 c3052: - aux_cap_helper__b6,r12,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b6,r12,band_HIGH_PAIRED__Z2 <= 0
 c3053: aux_cap_helper__b7,r9,band_UNPAIRED__condX1
        - aux_cap_helper__b7,r9,band_UNPAIRED__Z1 <= 0
 c3054: aux_cap_helper__b3,r2,band_UNPAIRED__Z2
        + aux_cap_helper__b3,r2,band_UNPAIRED__Z1  = 1
 c3055: - aux_cap_helper__b4,r2,band_LOW_PAIRED__condX1 <= 0
 c3056: - aux_cap_helper__b8,r5,band_LOW_PAIRED__Z1
        + aux_cap_helper__b8,r5,band_LOW_PAIRED__condX1 <= 0
 c3057: sv_function_helpervar_b7,r7__Z1 + sv_function_helpervar_b7,r7__Z2
        + sv_function_helpervar_b7,r7__Z3  = 1
 c3058: X__b8,r6,band_HIGH_PAIRED + X__b8,r6,band_UNPAIRED
        - GlobalBidder_W_ir__b8,r6 + X__b8,r6,band_LOW_PAIRED >= 0
 c3059: 2.0281668543 aux_cap_helper__b3,r7,band_HIGH_PAIRED__condX1
        + 2.8394335959 aux_cap_helper__b3,r7,band_HIGH_PAIRED__condX2
        + 1.8253501688 aux_cap_helper__b3,r7,band_HIGH_PAIRED__condX3
        - aux_cap_b3,r7,band_HIGH_PAIRED
        + 1.21690011250031 aux_cap_helper__b3,r7,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b3,r7,band_HIGH_PAIRED__Z2  = 0
 c3060: - 15.8453086887446 sv_function_helpervar_b4,r0__Z3
        + sv_function_helpervar_b4,r0__condX3 <= 0
 c3061: - aux_cap_helper__b6,r12,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b6,r12,band_HIGH_PAIRED__Z3 <= 0
 c3062: aux_cap_helper__b0,r3,band_LOW_PAIRED__condX1
        + aux_cap_helper__b0,r3,band_LOW_PAIRED__condX2
        - X__b0,r3,band_LOW_PAIRED  = 0
 c3063: aux_cap_helper__b7,r9,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b7,r9,band_UNPAIRED__Z2 <= 0
 c3064: - aux_cap_b8,r11,band_UNPAIRED
        + 0.888446608 aux_cap_helper__b8,r11,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b8,r11,band_UNPAIRED__condX1
        - 0.253841888 aux_cap_helper__b8,r11,band_UNPAIRED__Z2  = 0
 c3065: 2.0281668543 aux_cap_helper__b9,r7,band_HIGH_PAIRED__condX1
        + 2.8394335959 aux_cap_helper__b9,r7,band_HIGH_PAIRED__condX2
        + 1.8253501688 aux_cap_helper__b9,r7,band_HIGH_PAIRED__condX3
        + 1.21690011250031 aux_cap_helper__b9,r7,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b9,r7,band_HIGH_PAIRED__Z2
        - aux_cap_b9,r7,band_HIGH_PAIRED  = 0
 c3066: sv_function_helpervar_b9,r3__condX1
        - 0.158453086900186 sv_function_helpervar_b9,r3__Z1 <= 0
 c3067: sv_function_helpervar_b5,r8__condX1
        - 0.158453086900078 sv_function_helpervar_b5,r8__Z1 <= 0
 c3068: aux_cap_helper__b8,r5,band_UNPAIRED__Z2
        - aux_cap_helper__b8,r5,band_UNPAIRED__condX2 <= 0
 c3069: 2.0281668543 aux_cap_helper__b3,r2,band_HIGH_PAIRED__condX1
        + 2.8394335959 aux_cap_helper__b3,r2,band_HIGH_PAIRED__condX2
        + 1.8253501688 aux_cap_helper__b3,r2,band_HIGH_PAIRED__condX3
        - 0.8112667416 aux_cap_helper__b3,r2,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b3,r2,band_HIGH_PAIRED__Z3
        - aux_cap_b3,r2,band_HIGH_PAIRED  = 0
 c3070: - 0.253841888 aux_cap_helper__b9,r11,band_UNPAIRED__Z2
        + 0.63460472 aux_cap_helper__b9,r11,band_UNPAIRED__condX1
        + 0.888446608 aux_cap_helper__b9,r11,band_UNPAIRED__condX2
        - aux_cap_b9,r11,band_UNPAIRED  = 0
 c3071: aux_cap_helper__b8,r2,band_LOW_PAIRED__Z2
        + aux_cap_helper__b8,r2,band_LOW_PAIRED__Z1  = 1
 c3072: - aux_cap_helper__b9,r1,band_UNPAIRED__condX1 <= 0
 c3073: aux_cap_helper__b0,r13,band_UNPAIRED__Z1
        + aux_cap_helper__b0,r13,band_UNPAIRED__Z2  = 1
 c3074: - aux_cap_helper__b2,r3,band_LOW_PAIRED__Z1
        + aux_cap_helper__b2,r3,band_LOW_PAIRED__condX1 <= 0
 c3075: aux_cap_helper__b3,r8,band_UNPAIRED__condX1
        - aux_cap_helper__b3,r8,band_UNPAIRED__Z1 <= 0
 c3076: - 0.158453086898492 sv_function_helpervar_b0,r11__Z1
        + sv_function_helpervar_b0,r11__condX1 <= 0
 c3077: aux_cap_helper__b9,r11,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b9,r11,band_HIGH_PAIRED__Z1 <= 0
 c3078: - aux_cap_b2,r10,band_LOW_PAIRED
        - 1.2715511236 aux_cap_helper__b2,r10,band_LOW_PAIRED__Z2
        + 4.4504289328 aux_cap_helper__b2,r10,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b2,r10,band_LOW_PAIRED__condX1  = 0
 c3079: aux_cap_helper__b1,r7,band_UNPAIRED__Z1
        + aux_cap_helper__b1,r7,band_UNPAIRED__Z2  = 1
 c3080: - aux_cap_helper__b7,r3,band_LOW_PAIRED__condX2
        + aux_cap_helper__b7,r3,band_LOW_PAIRED__Z2 <= 0
 c3081: - 0.253841888 aux_cap_helper__b7,r11,band_UNPAIRED__Z2
        - aux_cap_b7,r11,band_UNPAIRED
        + 0.63460472 aux_cap_helper__b7,r11,band_UNPAIRED__condX1
        + 0.888446608 aux_cap_helper__b7,r11,band_UNPAIRED__condX2  = 0
 c3082: - 2 aux_cap_helper__b6,r13,band_UNPAIRED__Z2
        + aux_cap_helper__b6,r13,band_UNPAIRED__condX2 <= 0
 c3083: aux_cap_helper__b9,r6,band_UNPAIRED__condX1
        + aux_cap_helper__b9,r6,band_UNPAIRED__condX2 - X__b9,r6,band_UNPAIRED
         = 0
 c3084: - 2 aux_cap_helper__b4,r12,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b4,r12,band_HIGH_PAIRED__condX2 <= 0
 c3085: - sv_function_helpervar_b8,r3__condX1 <= 0
 c3086: - aux_cap_helper__b9,r1,band_LOW_PAIRED__Z1
        + aux_cap_helper__b9,r1,band_LOW_PAIRED__condX1 <= 0
 c3087: aux_cap_helper__b3,r7,band_LOW_PAIRED__Z2
        + aux_cap_helper__b3,r7,band_LOW_PAIRED__Z1  = 1
 c3088: aux_cap_helper__b0,r13,band_LOW_PAIRED__Z2
        - aux_cap_helper__b0,r13,band_LOW_PAIRED__condX2 <= 0
 c3089: aux_cap_b1,r1,band_UNPAIRED + aux_cap_b1,r1,band_HIGH_PAIRED
        + aux_cap_b1,r1,band_LOW_PAIRED - aux_c_b1,r1  = 0
 c3090: - aux_c_b7,r13 + aux_cap_b7,r13,band_LOW_PAIRED
        + aux_cap_b7,r13,band_HIGH_PAIRED + aux_cap_b7,r13,band_UNPAIRED  = 0
 c3091: sv_function_helpervar_b8,r2__condX1
        - 0.158453086941469 sv_function_helpervar_b8,r2__Z1 <= 0
 c3092: - 2 aux_cap_helper__b9,r1,band_LOW_PAIRED__Z2
        + aux_cap_helper__b9,r1,band_LOW_PAIRED__condX2 <= 0
 c3093: aux_cap_b3,r10,band_LOW_PAIRED + aux_cap_b3,r10,band_UNPAIRED
        - aux_c_b3,r10 + aux_cap_b3,r10,band_HIGH_PAIRED  = 0
 c3094: aux_cap_helper__b6,r3,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b6,r3,band_HIGH_PAIRED__condX2 <= 0
 c3095: - aux_cap_helper__b1,r11,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b1,r11,band_HIGH_PAIRED__Z2 <= 0
 c3096: 4.4504289328 aux_cap_helper__b6,r0,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b6,r0,band_LOW_PAIRED__condX1
        - 1.2715511236 aux_cap_helper__b6,r0,band_LOW_PAIRED__Z2
        - aux_cap_b6,r0,band_LOW_PAIRED  = 0
 c3097: - aux_cap_helper__b7,r6,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b7,r6,band_HIGH_PAIRED__Z2 <= 0
 c3098: 51.8352123236942 aux_sv_b8,r13 - aux_Omega_b8,r13  = 0
 c3099: - 15.8453086887446 sv_function_helpervar_b6,r5__Z3
        + sv_function_helpervar_b6,r5__condX3 <= 0
 c3100: - aux_cap_helper__b9,r8,band_LOW_PAIRED__Z1
        + aux_cap_helper__b9,r8,band_LOW_PAIRED__condX1 <= 0
 c3101: aux_Omega_b0,r12 + aux_Omega_b0,r10 - v__0 + aux_Omega_b0,r6
        + aux_Omega_b0,r7  = 0
 c3102: 2 aux_cap_helper__b6,r3,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b6,r3,band_HIGH_PAIRED__condX3 <= 0
 c3103: - aux_cap_helper__b1,r11,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b1,r11,band_HIGH_PAIRED__Z3 <= 0
 c3104: - aux_cap_helper__b7,r6,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b7,r6,band_HIGH_PAIRED__Z3 <= 0
 c3105: - aux_cap_helper__b8,r2,band_LOW_PAIRED__Z1
        + aux_cap_helper__b8,r2,band_LOW_PAIRED__condX1 <= 0
 c3106: - aux_cap_helper__b8,r9,band_UNPAIRED__Z1
        + aux_cap_helper__b8,r9,band_UNPAIRED__condX1 <= 0
 c3107: - aux_cap_helper__b1,r11,band_UNPAIRED__condX2
        + aux_cap_helper__b1,r11,band_UNPAIRED__Z2 <= 0
 c3108: - 0.158453086934315 sv_function_helpervar_b2,r11__Z1
        + sv_function_helpervar_b2,r11__condX1 <= 0
 c3109: - aux_cap_helper__b6,r13,band_UNPAIRED__Z1
        + aux_cap_helper__b6,r13,band_UNPAIRED__condX1 <= 0
 c3110: - aux_cap_helper__b0,r12,band_UNPAIRED__condX1 <= 0
 c3111: - 1.2715511236 aux_cap_helper__b1,r10,band_LOW_PAIRED__Z2
        - aux_cap_b1,r10,band_LOW_PAIRED
        + 4.4504289328 aux_cap_helper__b1,r10,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b1,r10,band_LOW_PAIRED__condX1  = 0
 c3112: sv_function_helpervar_b1,r10__Z3 + sv_function_helpervar_b1,r10__Z2
        + sv_function_helpervar_b1,r10__Z1  = 1
 c3113: - aux_Omega_b4,r6 + 42.9111846907184 aux_sv_b4,r6  = 0
 c3114: - sv_function_helpervar_b5,r4__condX3
        + 5.55364789890483 sv_function_helpervar_b5,r4__Z3 <= 0
 c3115: 0.0102040816326531 X__b8,r4,band_HIGH_PAIRED
        + 0.0102040816326531 X__b8,r4,band_UNPAIRED
        + 0.0102040816326531 X__b8,r4,band_LOW_PAIRED
        - GlobalBidder_W_ir__b8,r4 <= 0
 c3116: - aux_cap_helper__b0,r0,band_LOW_PAIRED__Z1
        + aux_cap_helper__b0,r0,band_LOW_PAIRED__condX1 <= 0
 c3117: - sv_function_helpervar_b1,r11__condX1 <= 0
 c3118: - sv_function_helpervar_b9,r11__condX2
        + 0.158453086979881 sv_function_helpervar_b9,r11__Z2 <= 0
 c3119: aux_cap_helper__b2,r4,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b2,r4,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b2,r4,band_HIGH_PAIRED__Z3  = 1
 c3120: aux_cap_helper__b0,r6,band_UNPAIRED__Z1
        + aux_cap_helper__b0,r6,band_UNPAIRED__Z2  = 1
 c3121: - X__b6,r5,band_UNPAIRED + aux_cap_helper__b6,r5,band_UNPAIRED__condX1
        + aux_cap_helper__b6,r5,band_UNPAIRED__condX2  = 0
 c3122: - sv_function_helpervar_b8,r8__condX3
        + 6.47080441264293 sv_function_helpervar_b8,r8__Z3 <= 0
 c3123: aux_cap_helper__b0,r8,band_LOW_PAIRED__Z1
        + aux_cap_helper__b0,r8,band_LOW_PAIRED__Z2  = 1
 c3124: - aux_cap_helper__b1,r8,band_UNPAIRED__condX2
        + aux_cap_helper__b1,r8,band_UNPAIRED__Z2 <= 0
 c3125: - aux_cap_helper__b1,r0,band_UNPAIRED__condX1 <= 0
 c3126: aux_cap_helper__b3,r7,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b3,r7,band_HIGH_PAIRED__Z2 <= 0
 c3127: aux_cap_helper__b5,r7,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b5,r7,band_UNPAIRED__Z2 <= 0
 c3128: 6.15627515730785 sv_function_helpervar_b5,r13__Z3
        - sv_function_helpervar_b5,r13__condX3 <= 0
 c3129: aux_cap_helper__b4,r2,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b4,r2,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b4,r2,band_HIGH_PAIRED__Z2  = 1
 c3130: - sv_function_helpervar_b4,r8__condX1 <= 0
 c3131: - 2 aux_cap_helper__b0,r0,band_LOW_PAIRED__Z2
        + aux_cap_helper__b0,r0,band_LOW_PAIRED__condX2 <= 0
 c3132: - 3 aux_cap_helper__b6,r9,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b6,r9,band_HIGH_PAIRED__condX3 <= 0
 c3133: aux_cap_helper__b5,r12,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b5,r12,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b5,r12,band_HIGH_PAIRED__Z1  = 1
 c3134: aux_cap_helper__b9,r12,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b9,r12,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b9,r12,band_HIGH_PAIRED__condX1
        - X__b9,r12,band_HIGH_PAIRED  = 0
 c3135: aux_cap_b2,r7,band_UNPAIRED - aux_c_b2,r7
        + aux_cap_b2,r7,band_HIGH_PAIRED + aux_cap_b2,r7,band_LOW_PAIRED  = 0
 c3136: aux_cap_helper__b2,r3,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b2,r3,band_HIGH_PAIRED__Z1 <= 0
 c3137: aux_cap_helper__b6,r0,band_UNPAIRED__Z1
        + aux_cap_helper__b6,r0,band_UNPAIRED__Z2  = 1
 c3138: - aux_cap_helper__b7,r4,band_UNPAIRED__condX2
        + aux_cap_helper__b7,r4,band_UNPAIRED__Z2 <= 0
 c3139: 142.450622514205 sv_function_helpervar_b6,r8__Z2
        + 312.312334227586 sv_function_helpervar_b6,r8__Z3
        + 15.1947645047 sv_function_helpervar_b6,r8__condX3
        + 942.4308564261 sv_function_helpervar_b6,r8__condX1
        + 43.4226688033 sv_function_helpervar_b6,r8__condX2 - aux_sv_b6,r8  = 0
 c3140: - X__b8,r11,band_HIGH_PAIRED
        + aux_cap_helper__b8,r11,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b8,r11,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b8,r11,band_HIGH_PAIRED__condX2  = 0
 c3141: 0.158453086893361 sv_function_helpervar_b5,r6__Z2
        - sv_function_helpervar_b5,r6__condX2 <= 0
 c3142: - 0.158453086927404 sv_function_helpervar_b0,r8__Z1
        + sv_function_helpervar_b0,r8__condX1 <= 0
 c3143: aux_Omega_b7,r10 + aux_Omega_b7,r11 + aux_Omega_b7,r12
        + aux_Omega_b7,r13 + aux_Omega_b7,r0 + aux_Omega_b7,r1
        + aux_Omega_b7,r2 + aux_Omega_b7,r3 + aux_Omega_b7,r4 + aux_Omega_b7,r5
        + aux_Omega_b7,r6 + aux_Omega_b7,r7 + aux_Omega_b7,r8 + aux_Omega_b7,r9
        - GlobalBidder_PSI__i7,k1 - 20668026.8748422 GlobalBidder_hat{W}__i7,k1
        >= -20668026.8748422
 c3144: 1995.2925269408 sv_function_helpervar_b9,r10__condX1
        + 80.1993849008 sv_function_helpervar_b9,r10__condX2
        + 35.2442363829 sv_function_helpervar_b9,r10__condX3 - aux_sv_b9,r10
        + 612.508121650269 sv_function_helpervar_b9,r10__Z3
        + 303.452420085379 sv_function_helpervar_b9,r10__Z2  = 0
 c3145: - aux_cap_helper__b1,r6,band_LOW_PAIRED__condX1 <= 0
 c3146: - sv_function_helpervar_b1,r6__condX3
        + 6.07000388758382 sv_function_helpervar_b1,r6__Z3 <= 0
 c3147: - aux_cap_helper__b5,r0,band_HIGH_PAIRED__condX1 <= 0
 c3148: - aux_cap_helper__b1,r1,band_HIGH_PAIRED__condX1 <= 0
 c3149: - sv_function_helpervar_b9,r1__condX1 <= 0
 c3150: - sv_function_helpervar_b2,r3__condX3
        + 6.30553906524748 sv_function_helpervar_b2,r3__Z3 <= 0
 c3151: - sv_function_helpervar_b6,r7__condX1 <= 0
 c3152: - 2 aux_cap_helper__b9,r8,band_LOW_PAIRED__Z2
        + aux_cap_helper__b9,r8,band_LOW_PAIRED__condX2 <= 0
 c3153: - sv_function_helpervar_b8,r11__condX2
        + 0.158453087021465 sv_function_helpervar_b8,r11__Z2 <= 0
 c3154: - aux_cap_helper__b6,r4,band_LOW_PAIRED__condX1 <= 0
 c3155: - sv_function_helpervar_b8,r13__condX1 <= 0
 c3156: - 2 aux_cap_helper__b8,r2,band_LOW_PAIRED__Z2
        + aux_cap_helper__b8,r2,band_LOW_PAIRED__condX2 <= 0
 c3157: aux_cap_helper__b3,r7,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b3,r7,band_HIGH_PAIRED__Z1 <= 0
 c3158: - aux_cap_helper__b4,r2,band_UNPAIRED__condX1 <= 0
 c3159: aux_cap_helper__b5,r7,band_UNPAIRED__condX1
        - aux_cap_helper__b5,r7,band_UNPAIRED__Z1 <= 0
 c3160: aux_cap_helper__b6,r2,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b6,r2,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b6,r2,band_HIGH_PAIRED__condX2
        - X__b6,r2,band_HIGH_PAIRED  = 0
 c3161: - 2 aux_cap_helper__b8,r9,band_UNPAIRED__Z2
        + aux_cap_helper__b8,r9,band_UNPAIRED__condX2 <= 0
 c3162: - aux_cap_helper__b4,r9,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b4,r9,band_HIGH_PAIRED__Z2 <= 0
 c3163: - 6.99600389083181 sv_function_helpervar_b7,r8__Z2
        + sv_function_helpervar_b7,r8__condX2 <= 0
 c3164: sv_function_helpervar_b7,r7__condX2
        - 8.08713064043177 sv_function_helpervar_b7,r7__Z2 <= 0
 c3165: aux_cap_helper__b4,r3,band_UNPAIRED__Z1
        + aux_cap_helper__b4,r3,band_UNPAIRED__Z2  = 1
 c3166: - sv_function_helpervar_b6,r1__condX2
        + 0.158453086915037 sv_function_helpervar_b6,r1__Z2 <= 0
 c3167: 2362.23013399777 aux_sv_b0,r7 - aux_Omega_b0,r7  = 0
 c3168: aux_cap_helper__b6,r3,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b6,r3,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b6,r3,band_HIGH_PAIRED__Z1  = 1
 c3169: sv_function_helpervar_b9,r7__condX1
        - 0.158453086895241 sv_function_helpervar_b9,r7__Z1 <= 0
 c3170: - sv_function_helpervar_b7,r2__condX1 <= 0
 c3171: 5.78349746755236 sv_function_helpervar_b1,r3__Z3
        - sv_function_helpervar_b1,r3__condX3 <= 0
 c3172: - aux_cap_helper__b4,r9,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b4,r9,band_HIGH_PAIRED__Z3 <= 0
 c3173: 0.158453086911987 sv_function_helpervar_b4,r3__Z2
        - sv_function_helpervar_b4,r3__condX2 <= 0
 c3174: aux_cap_helper__b9,r13,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b9,r13,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b9,r13,band_HIGH_PAIRED__Z2  = 1
 c3175: - 0.253841888 aux_cap_helper__b4,r13,band_UNPAIRED__Z2
        + 0.888446608 aux_cap_helper__b4,r13,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b4,r13,band_UNPAIRED__condX1
        - aux_cap_b4,r13,band_UNPAIRED  = 0
 c3176: aux_cap_helper__b1,r12,band_LOW_PAIRED__Z2
        - aux_cap_helper__b1,r12,band_LOW_PAIRED__condX2 <= 0
 c3177: - sv_function_helpervar_b2,r4__condX2
        + 0.158453086899697 sv_function_helpervar_b2,r4__Z2 <= 0
 c3178: - 6.21227438384747 sv_function_helpervar_b3,r2__Z2
        + sv_function_helpervar_b3,r2__condX2 <= 0
 c3179: aux_cap_helper__b0,r4,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b0,r4,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b0,r4,band_HIGH_PAIRED__condX2
        - X__b0,r4,band_HIGH_PAIRED  = 0
 c3180: - aux_cap_helper__b2,r8,band_LOW_PAIRED__condX1 <= 0
 c3181: 23.7956663952 sv_function_helpervar_b7,r13__condX3
        + 46.2737284962 sv_function_helpervar_b7,r13__condX2
        + 1255.6694400557 sv_function_helpervar_b7,r13__condX1
        + 191.632483783287 sv_function_helpervar_b7,r13__Z2
        + 359.856612758969 sv_function_helpervar_b7,r13__Z3 - aux_sv_b7,r13
         = 0
 c3182: aux_cap_b9,r1,band_HIGH_PAIRED + aux_cap_b9,r1,band_LOW_PAIRED
        + aux_cap_b9,r1,band_UNPAIRED - aux_c_b9,r1  = 0
 c3183: sv_function_helpervar_b2,r10__condX3
        - 15.8453086887446 sv_function_helpervar_b2,r10__Z3 <= 0
 c3184: - 1.2715511236 aux_cap_helper__b9,r4,band_LOW_PAIRED__Z2
        - aux_cap_b9,r4,band_LOW_PAIRED
        + 4.4504289328 aux_cap_helper__b9,r4,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b9,r4,band_LOW_PAIRED__condX1  = 0
 c3185: aux_cap_helper__b0,r7,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b0,r7,band_LOW_PAIRED__Z2 <= 0
 c3186: 6.6326479692379 sv_function_helpervar_b1,r1__Z3
        - sv_function_helpervar_b1,r1__condX3 <= 0
 c3187: aux_cap_helper__b8,r11,band_UNPAIRED__condX2
        + aux_cap_helper__b8,r11,band_UNPAIRED__condX1
        - X__b8,r11,band_UNPAIRED  = 0
 c3188: - 2 aux_cap_helper__b3,r6,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b3,r6,band_HIGH_PAIRED__condX2 <= 0
 c3189: - sv_function_helpervar_b4,r2__condX2
        + 0.158453086926367 sv_function_helpervar_b4,r2__Z2 <= 0
 c3190: - aux_cap_helper__b4,r11,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b4,r11,band_HIGH_PAIRED__Z2 <= 0
 c3191: - sv_function_helpervar_b4,r5__condX3
        + 5.81826105166577 sv_function_helpervar_b4,r5__Z3 <= 0
 c3192: - X__b8,r7,band_UNPAIRED + aux_cap_helper__b8,r7,band_UNPAIRED__condX1
        + aux_cap_helper__b8,r7,band_UNPAIRED__condX2  = 0
 c3193: aux_cap_helper__b9,r10,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b9,r10,band_HIGH_PAIRED__condX2 <= 0
 c3194: - aux_cap_helper__b9,r9,band_UNPAIRED__condX1 <= 0
 c3195: 2 aux_cap_helper__b9,r0,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b9,r0,band_HIGH_PAIRED__condX3 <= 0
 c3196: 34.9818650686465 aux_sv_b2,r13 - aux_Omega_b2,r13  = 0
 c3197: 0.0102040816326531 X__b9,r12,band_LOW_PAIRED
        - GlobalBidder_W_ir__b9,r12
        + 0.0102040816326531 X__b9,r12,band_UNPAIRED
        + 0.0102040816326531 X__b9,r12,band_HIGH_PAIRED <= 0
 c3198: - aux_cap_helper__b4,r11,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b4,r11,band_HIGH_PAIRED__Z3 <= 0
 c3199: 2 aux_cap_helper__b9,r10,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b9,r10,band_HIGH_PAIRED__condX3 <= 0
 c3200: - aux_cap_helper__b0,r9,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b0,r9,band_HIGH_PAIRED__condX1 <= 0
 c3201: sv_function_helpervar_b6,r1__condX3
        + sv_function_helpervar_b6,r1__condX2
        + sv_function_helpervar_b6,r1__condX1 - aux_c_b6,r1  = 0
 c3202: - sv_function_helpervar_b5,r9__condX1 <= 0
 c3203: 3.1788778092 aux_cap_helper__b5,r6,band_LOW_PAIRED__condX1
        + 4.4504289328 aux_cap_helper__b5,r6,band_LOW_PAIRED__condX2
        - aux_cap_b5,r6,band_LOW_PAIRED
        - 1.2715511236 aux_cap_helper__b5,r6,band_LOW_PAIRED__Z2  = 0
 c3204: - aux_cap_helper__b2,r10,band_UNPAIRED__condX1 <= 0
 c3205: - aux_cap_helper__b3,r5,band_UNPAIRED__condX2
        + aux_cap_helper__b3,r5,band_UNPAIRED__Z2 <= 0
 c3206: aux_cap_helper__b9,r7,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b9,r7,band_HIGH_PAIRED__Z3 <= 0
 c3207: - aux_cap_helper__b4,r3,band_LOW_PAIRED__condX2
        + aux_cap_helper__b4,r3,band_LOW_PAIRED__Z2 <= 0
 c3208: - 3 aux_cap_helper__b3,r0,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b3,r0,band_HIGH_PAIRED__condX3 <= 0
 c3209: aux_cap_helper__b7,r3,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b7,r3,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b7,r3,band_HIGH_PAIRED__Z3  = 1
 c3210: sv_function_helpervar_b2,r5__Z2 + sv_function_helpervar_b2,r5__Z3
        + sv_function_helpervar_b2,r5__Z1  = 1
 c3211: - aux_cap_helper__b5,r1,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b5,r1,band_HIGH_PAIRED__condX1 <= 0
 c3212: sv_function_helpervar_b7,r6__condX1
        - 0.158453086898302 sv_function_helpervar_b7,r6__Z1 <= 0
 c3213: sv_function_helpervar_b0,r12__Z2 + sv_function_helpervar_b0,r12__Z3
        + sv_function_helpervar_b0,r12__Z1  = 1
 c3214: - aux_cap_helper__b7,r13,band_HIGH_PAIRED__condX1 <= 0
 c3215: - 2 aux_cap_helper__b5,r1,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b5,r1,band_HIGH_PAIRED__condX2 <= 0
 c3216: aux_cap_helper__b0,r7,band_LOW_PAIRED__condX1
        - aux_cap_helper__b0,r7,band_LOW_PAIRED__Z1 <= 0
 c3217: - aux_cap_helper__b5,r5,band_LOW_PAIRED__condX1 <= 0
 c3218: - 0.158453086897641 sv_function_helpervar_b0,r10__Z1
        + sv_function_helpervar_b0,r10__condX1 <= 0
 c3219: - aux_cap_helper__b3,r6,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b3,r6,band_HIGH_PAIRED__condX1 <= 0
 c3220: - aux_cap_helper__b4,r7,band_UNPAIRED__Z1
        + aux_cap_helper__b4,r7,band_UNPAIRED__condX1 <= 0
 c3221: aux_cap_helper__b7,r13,band_LOW_PAIRED__Z1
        + aux_cap_helper__b7,r13,band_LOW_PAIRED__Z2  = 1
 c3222: aux_Omega_b7,r10 + aux_Omega_b7,r11 + aux_Omega_b7,r12
        + aux_Omega_b7,r13 + aux_Omega_b7,r0 + aux_Omega_b7,r1
        + aux_Omega_b7,r2 + aux_Omega_b7,r3 + aux_Omega_b7,r4 + aux_Omega_b7,r5
        + aux_Omega_b7,r6 + aux_Omega_b7,r7 + aux_Omega_b7,r8 + aux_Omega_b7,r9
        - GlobalBidder_PSI__i7,k2 + 20668026.8748422 GlobalBidder_hat{W}__i7,k2
        <= 20668026.8748422
 c3223: - 0.253841888 aux_cap_helper__b6,r1,band_UNPAIRED__Z2
        - aux_cap_b6,r1,band_UNPAIRED
        + 0.63460472 aux_cap_helper__b6,r1,band_UNPAIRED__condX1
        + 0.888446608 aux_cap_helper__b6,r1,band_UNPAIRED__condX2  = 0
 c3224: 1203.77356670394 aux_sv_b9,r12 - aux_Omega_b9,r12  = 0
 c3225: sv_function_helpervar_b0,r2__condX2
        - 5.87408986601521 sv_function_helpervar_b0,r2__Z2 <= 0
 c3226: aux_cap_helper__b1,r5,band_LOW_PAIRED__condX1
        + aux_cap_helper__b1,r5,band_LOW_PAIRED__condX2
        - X__b1,r5,band_LOW_PAIRED  = 0
 c3227: - aux_cap_helper__b2,r4,band_HIGH_PAIRED__condX1 <= 0
 c3228: aux_cap_helper__b9,r0,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b9,r0,band_HIGH_PAIRED__condX2 <= 0
 c3229: aux_cap_helper__b8,r5,band_LOW_PAIRED__Z2
        + aux_cap_helper__b8,r5,band_LOW_PAIRED__Z1  = 1
 c3230: - sv_function_helpervar_b0,r12__condX2
        + 0.158453086902941 sv_function_helpervar_b0,r12__Z2 <= 0
 c3231: X__b9,r8,band_HIGH_PAIRED - GlobalBidder_W_ir__b9,r8
        + X__b9,r8,band_UNPAIRED + X__b9,r8,band_LOW_PAIRED >= 0
 c3232: aux_cap_helper__b8,r13,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b8,r13,band_UNPAIRED__Z2 <= 0
 c3233: - 2 aux_cap_helper__b5,r9,band_LOW_PAIRED__Z2
        + aux_cap_helper__b5,r9,band_LOW_PAIRED__condX2 <= 0
 c3234: - sv_function_helpervar_b8,r0__condX1 <= 0
 c3235: aux_cap_helper__b4,r10,band_LOW_PAIRED__Z2
        - aux_cap_helper__b4,r10,band_LOW_PAIRED__condX2 <= 0
 c3236: - X__b6,r1,band_UNPAIRED + aux_cap_helper__b6,r1,band_UNPAIRED__condX1
        + aux_cap_helper__b6,r1,band_UNPAIRED__condX2  = 0
 c3237: aux_cap_helper__b0,r0,band_UNPAIRED__condX1
        - aux_cap_helper__b0,r0,band_UNPAIRED__Z1 <= 0
 c3238: - aux_cap_b2,r5,band_LOW_PAIRED
        - 1.2715511236 aux_cap_helper__b2,r5,band_LOW_PAIRED__Z2
        + 3.1788778092 aux_cap_helper__b2,r5,band_LOW_PAIRED__condX1
        + 4.4504289328 aux_cap_helper__b2,r5,band_LOW_PAIRED__condX2  = 0
 c3239: aux_cap_helper__b6,r8,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b6,r8,band_LOW_PAIRED__Z2 <= 0
 c3240: sv_function_helpervar_b0,r3__condX2
        + sv_function_helpervar_b0,r3__condX3
        + sv_function_helpervar_b0,r3__condX1 - aux_c_b0,r3  = 0
 c3241: sv_function_helpervar_b1,r0__Z2 + sv_function_helpervar_b1,r0__Z3
        + sv_function_helpervar_b1,r0__Z1  = 1
 c3242: 978.9018730695 sv_function_helpervar_b5,r8__condX1
        + 15.3003675424 sv_function_helpervar_b5,r8__condX3
        + 47.6216465231 sv_function_helpervar_b5,r8__condX2 - aux_sv_b5,r8
        + 147.564226665281 sv_function_helpervar_b5,r8__Z2
        + 332.042521982019 sv_function_helpervar_b5,r8__Z3  = 0
 c3243: - aux_Omega_b9,r2 + 447.524507135158 aux_sv_b9,r2  = 0
 c3244: sv_function_helpervar_b7,r2__condX2
        - 8.2076340013385 sv_function_helpervar_b7,r2__Z2 <= 0
 c3245: - aux_cap_helper__b3,r0,band_UNPAIRED__condX1 <= 0
 c3246: - aux_cap_helper__b2,r8,band_UNPAIRED__Z1
        + aux_cap_helper__b2,r8,band_UNPAIRED__condX1 <= 0
 c3247: aux_cap_helper__b3,r2,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b3,r2,band_HIGH_PAIRED__Z3 <= 0
 c3248: - X__b2,r5,band_HIGH_PAIRED
        + aux_cap_helper__b2,r5,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b2,r5,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b2,r5,band_HIGH_PAIRED__condX3  = 0
 c3249: aux_cap_helper__b0,r0,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b0,r0,band_UNPAIRED__Z2 <= 0
 c3250: - X__b4,r12,band_UNPAIRED
        + aux_cap_helper__b4,r12,band_UNPAIRED__condX1
        + aux_cap_helper__b4,r12,band_UNPAIRED__condX2  = 0
 c3251: aux_cap_helper__b8,r11,band_UNPAIRED__condX1
        - aux_cap_helper__b8,r11,band_UNPAIRED__Z1 <= 0
 c3252: sv_function_helpervar_b9,r6__condX2
        + sv_function_helpervar_b9,r6__condX3
        + sv_function_helpervar_b9,r6__condX1 - aux_c_b9,r6  = 0
 c3253: - aux_cap_helper__b6,r10,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b6,r10,band_HIGH_PAIRED__Z2 <= 0
 c3254: aux_cap_helper__b1,r0,band_LOW_PAIRED__Z2
        + aux_cap_helper__b1,r0,band_LOW_PAIRED__Z1  = 1
 c3255: - 2 aux_cap_helper__b2,r8,band_UNPAIRED__Z2
        + aux_cap_helper__b2,r8,band_UNPAIRED__condX2 <= 0
 c3256: 0.158453086921147 sv_function_helpervar_b2,r13__Z2
        - sv_function_helpervar_b2,r13__condX2 <= 0
 c3257: X__b4,r0,band_UNPAIRED + X__b3,r0,band_UNPAIRED
        + X__b5,r0,band_UNPAIRED + X__b1,r0,band_UNPAIRED
        + X__b7,r0,band_UNPAIRED + X__b2,r0,band_UNPAIRED
        + X__b6,r0,band_UNPAIRED + X__b0,r0,band_UNPAIRED
        + X__b8,r0,band_UNPAIRED + X__b9,r0,band_UNPAIRED <= 2
 c3258: - aux_cap_helper__b3,r5,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b3,r5,band_HIGH_PAIRED__Z2 <= 0
 c3259: aux_cap_helper__b7,r6,band_LOW_PAIRED__Z2
        - aux_cap_helper__b7,r6,band_LOW_PAIRED__condX2 <= 0
 c3260: 22.8556180346655 sv_function_helpervar_b1,r5__Z2
        + 49.1476870088427 sv_function_helpervar_b1,r5__Z3
        + 2.4889335858 sv_function_helpervar_b1,r5__condX3
        + 6.7054302393 sv_function_helpervar_b1,r5__condX2
        + 150.9476061582 sv_function_helpervar_b1,r5__condX1 - aux_sv_b1,r5
         = 0
 c3261: aux_cap_helper__b8,r11,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b8,r11,band_UNPAIRED__Z2 <= 0
 c3262: X__b5,r11,band_HIGH_PAIRED + X__b2,r11,band_HIGH_PAIRED
        + X__b0,r11,band_HIGH_PAIRED + X__b4,r11,band_HIGH_PAIRED
        + X__b3,r11,band_HIGH_PAIRED + X__b8,r11,band_HIGH_PAIRED
        + X__b6,r11,band_HIGH_PAIRED + X__b7,r11,band_HIGH_PAIRED
        + X__b9,r11,band_HIGH_PAIRED + X__b1,r11,band_HIGH_PAIRED <= 3
 c3263: - X__b0,r3,band_UNPAIRED + aux_cap_helper__b0,r3,band_UNPAIRED__condX2
        + aux_cap_helper__b0,r3,band_UNPAIRED__condX1  = 0
 c3264: aux_cap_helper__b8,r13,band_UNPAIRED__condX1
        - aux_cap_helper__b8,r13,band_UNPAIRED__Z1 <= 0
 c3265: - aux_cap_helper__b6,r10,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b6,r10,band_HIGH_PAIRED__Z3 <= 0
 c3266: - aux_c_b4,r1 + sv_function_helpervar_b4,r1__condX1
        + sv_function_helpervar_b4,r1__condX2
        + sv_function_helpervar_b4,r1__condX3  = 0
 c3267: - 5.68383285585204 sv_function_helpervar_b5,r11__Z2
        + sv_function_helpervar_b5,r11__condX2 <= 0
 c3268: aux_cap_helper__b7,r6,band_LOW_PAIRED__Z1
        + aux_cap_helper__b7,r6,band_LOW_PAIRED__Z2  = 1
 c3269: aux_cap_helper__b9,r4,band_UNPAIRED__Z2
        + aux_cap_helper__b9,r4,band_UNPAIRED__Z1  = 1
 c3270: aux_cap_helper__b5,r9,band_UNPAIRED__Z2
        + aux_cap_helper__b5,r9,band_UNPAIRED__Z1  = 1
 c3271: - aux_cap_helper__b4,r10,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b4,r10,band_HIGH_PAIRED__condX1 <= 0
 c3272: aux_cap_helper__b6,r4,band_LOW_PAIRED__condX1
        + aux_cap_helper__b6,r4,band_LOW_PAIRED__condX2
        - X__b6,r4,band_LOW_PAIRED  = 0
 c3273: aux_cap_helper__b2,r11,band_LOW_PAIRED__Z2
        - aux_cap_helper__b2,r11,band_LOW_PAIRED__condX2 <= 0
 c3274: - aux_cap_helper__b5,r12,band_LOW_PAIRED__condX2
        + aux_cap_helper__b5,r12,band_LOW_PAIRED__Z2 <= 0
 c3275: aux_cap_helper__b6,r8,band_LOW_PAIRED__condX1
        - aux_cap_helper__b6,r8,band_LOW_PAIRED__Z1 <= 0
 c3276: - 2 aux_cap_helper__b9,r4,band_LOW_PAIRED__Z2
        + aux_cap_helper__b9,r4,band_LOW_PAIRED__condX2 <= 0
 c3277: - aux_cap_helper__b6,r2,band_UNPAIRED__condX1 <= 0
 c3278: 0.158453086953718 sv_function_helpervar_b5,r7__Z2
        - sv_function_helpervar_b5,r7__condX2 <= 0
 c3279: - sv_function_helpervar_b1,r6__condX1 <= 0
 c3280: - 2 aux_cap_helper__b4,r10,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b4,r10,band_HIGH_PAIRED__condX2 <= 0
 c3281: - aux_Omega_b5,r0 + 121.91498754173 aux_sv_b5,r0  = 0
 c3282: - 15.8453086887446 sv_function_helpervar_b0,r8__Z3
        + sv_function_helpervar_b0,r8__condX3 <= 0
 c3283: - aux_cap_b1,r3,band_UNPAIRED
        + 0.888446608 aux_cap_helper__b1,r3,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b1,r3,band_UNPAIRED__condX1
        - 0.253841888 aux_cap_helper__b1,r3,band_UNPAIRED__Z2  = 0
 c3284: aux_cap_helper__b9,r5,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b9,r5,band_LOW_PAIRED__Z2 <= 0
 c3285: aux_cap_b5,r7,band_UNPAIRED - aux_c_b5,r7
        + aux_cap_b5,r7,band_HIGH_PAIRED + aux_cap_b5,r7,band_LOW_PAIRED  = 0
 c3286: - X__b7,r0,band_UNPAIRED + aux_cap_helper__b7,r0,band_UNPAIRED__condX2
        + aux_cap_helper__b7,r0,band_UNPAIRED__condX1  = 0
 c3287: 1255.6694400569 sv_function_helpervar_b7,r2__condX1
        + 26.0504285728 sv_function_helpervar_b7,r2__condX3
        + 42.113215995 sv_function_helpervar_b7,r2__condX2
        + 192.291729832215 sv_function_helpervar_b7,r2__Z2
        + 324.129210034921 sv_function_helpervar_b7,r2__Z3 - aux_sv_b7,r2  = 0
 c3288: - aux_cap_helper__b8,r1,band_UNPAIRED__condX1 <= 0
 c3289: - aux_Omega_b1,r7 + 1543.35743312441 aux_sv_b1,r7  = 0
 c3290: - aux_cap_helper__b3,r10,band_LOW_PAIRED__condX2
        + aux_cap_helper__b3,r10,band_LOW_PAIRED__Z2 <= 0
 c3291: aux_cap_helper__b6,r11,band_LOW_PAIRED__Z2
        - aux_cap_helper__b6,r11,band_LOW_PAIRED__condX2 <= 0
 c3292: - aux_cap_helper__b6,r10,band_UNPAIRED__condX2
        + aux_cap_helper__b6,r10,band_UNPAIRED__Z2 <= 0
 c3293: sv_function_helpervar_b9,r11__condX1
        + sv_function_helpervar_b9,r11__condX3
        + sv_function_helpervar_b9,r11__condX2 - aux_c_b9,r11  = 0
 c3294: sv_function_helpervar_b4,r11__Z3 + sv_function_helpervar_b4,r11__Z1
        + sv_function_helpervar_b4,r11__Z2  = 1
 c3295: GlobalBidder_PSI__i9,k0 - 20668026.8748422 GlobalBidder_hat{W}__i9,k0
        <= 0
 c3296: - aux_sv_b4,r13 + 14.6119267942 sv_function_helpervar_b4,r13__condX3
        + 883.4028074625 sv_function_helpervar_b4,r13__condX1
        + 39.0494293702 sv_function_helpervar_b4,r13__condX2
        + 133.790399233282 sv_function_helpervar_b4,r13__Z2
        + 286.906182966713 sv_function_helpervar_b4,r13__Z3  = 0
 c3297: - aux_cap_b7,r0,band_LOW_PAIRED
        - 1.2715511236 aux_cap_helper__b7,r0,band_LOW_PAIRED__Z2
        + 3.1788778092 aux_cap_helper__b7,r0,band_LOW_PAIRED__condX1
        + 4.4504289328 aux_cap_helper__b7,r0,band_LOW_PAIRED__condX2  = 0
 c3298: 0.158453086915011 sv_function_helpervar_b0,r0__Z2
        - sv_function_helpervar_b0,r0__condX2 <= 0
 c3299: sv_function_helpervar_b3,r9__condX1
        - 0.158453086901293 sv_function_helpervar_b3,r9__Z1 <= 0
 c3300: sv_function_helpervar_b4,r2__condX2
        - 6.17498435770839 sv_function_helpervar_b4,r2__Z2 <= 0
 c3301: - aux_cap_helper__b6,r9,band_UNPAIRED__condX1 <= 0
 c3302: - aux_cap_helper__b4,r9,band_LOW_PAIRED__condX2
        + aux_cap_helper__b4,r9,band_LOW_PAIRED__Z2 <= 0
 c3303: aux_cap_helper__b0,r10,band_LOW_PAIRED__Z2
        - aux_cap_helper__b0,r10,band_LOW_PAIRED__condX2 <= 0
 c3304: aux_cap_helper__b3,r11,band_LOW_PAIRED__Z2
        - aux_cap_helper__b3,r11,band_LOW_PAIRED__condX2 <= 0
 c3305: 0.888446608 aux_cap_helper__b3,r3,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b3,r3,band_UNPAIRED__condX1
        - aux_cap_b3,r3,band_UNPAIRED
        - 0.253841888 aux_cap_helper__b3,r3,band_UNPAIRED__Z2  = 0
 c3306: - 0.253841888 aux_cap_helper__b6,r5,band_UNPAIRED__Z2
        + 0.63460472 aux_cap_helper__b6,r5,band_UNPAIRED__condX1
        + 0.888446608 aux_cap_helper__b6,r5,band_UNPAIRED__condX2
        - aux_cap_b6,r5,band_UNPAIRED  = 0
 c3307: - 15.8453086887446 sv_function_helpervar_b4,r11__Z3
        + sv_function_helpervar_b4,r11__condX3 <= 0
 c3308: - sv_function_helpervar_b0,r3__condX2
        + 0.158453086889895 sv_function_helpervar_b0,r3__Z2 <= 0
 c3309: sv_function_helpervar_b2,r2__condX3
        - 15.8453086887446 sv_function_helpervar_b2,r2__Z3 <= 0
 c3310: aux_cap_b8,r11,band_LOW_PAIRED + aux_cap_b8,r11,band_UNPAIRED
        + aux_cap_b8,r11,band_HIGH_PAIRED - aux_c_b8,r11  = 0
 c3311: aux_cap_b9,r4,band_HIGH_PAIRED + aux_cap_b9,r4,band_UNPAIRED
        + aux_cap_b9,r4,band_LOW_PAIRED - aux_c_b9,r4  = 0
 c3312: - aux_cap_b9,r8,band_LOW_PAIRED
        - 1.2715511236 aux_cap_helper__b9,r8,band_LOW_PAIRED__Z2
        + 4.4504289328 aux_cap_helper__b9,r8,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b9,r8,band_LOW_PAIRED__condX1  = 0
 c3313: - aux_cap_helper__b9,r4,band_LOW_PAIRED__Z1
        + aux_cap_helper__b9,r4,band_LOW_PAIRED__condX1 <= 0
 c3314: 375.872380293662 sv_function_helpervar_b3,r2__Z3
        + 174.426490362393 sv_function_helpervar_b3,r2__Z2
        + 18.9522530264 sv_function_helpervar_b3,r2__condX3
        + 51.3793284394 sv_function_helpervar_b3,r2__condX2
        + 1152.1877364592 sv_function_helpervar_b3,r2__condX1 - aux_sv_b3,r2
         = 0
 c3315: aux_cap_helper__b4,r13,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b4,r13,band_HIGH_PAIRED__condX2 <= 0
 c3316: sv_function_helpervar_b6,r6__condX2
        - 5.93043276387834 sv_function_helpervar_b6,r6__Z2 <= 0
 c3317: aux_cap_helper__b9,r5,band_LOW_PAIRED__condX1
        - aux_cap_helper__b9,r5,band_LOW_PAIRED__Z1 <= 0
 c3318: 2 aux_cap_helper__b4,r13,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b4,r13,band_HIGH_PAIRED__condX3 <= 0
 c3319: X__b8,r5,band_HIGH_PAIRED + X__b1,r5,band_HIGH_PAIRED
        + X__b4,r5,band_HIGH_PAIRED + X__b9,r5,band_HIGH_PAIRED
        + X__b5,r5,band_HIGH_PAIRED + X__b7,r5,band_HIGH_PAIRED
        + X__b6,r5,band_HIGH_PAIRED + X__b2,r5,band_HIGH_PAIRED
        + X__b3,r5,band_HIGH_PAIRED + X__b0,r5,band_HIGH_PAIRED <= 3
 c3320: aux_cap_helper__b2,r6,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b2,r6,band_HIGH_PAIRED__Z2 <= 0
 c3321: - aux_cap_helper__b0,r8,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b0,r8,band_HIGH_PAIRED__Z2 <= 0
 c3322: - 15.8453086887446 sv_function_helpervar_b3,r2__Z3
        + sv_function_helpervar_b3,r2__condX3 <= 0
 c3323: aux_cap_helper__b2,r12,band_LOW_PAIRED__Z2
        - aux_cap_helper__b2,r12,band_LOW_PAIRED__condX2 <= 0
 c3324: - 3 aux_cap_helper__b9,r5,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b9,r5,band_HIGH_PAIRED__condX3 <= 0
 c3325: - X__b5,r9,band_HIGH_PAIRED
        + aux_cap_helper__b5,r9,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b5,r9,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b5,r9,band_HIGH_PAIRED__condX1  = 0
 c3326: - X__b2,r11,band_UNPAIRED
        + aux_cap_helper__b2,r11,band_UNPAIRED__condX2
        + aux_cap_helper__b2,r11,band_UNPAIRED__condX1  = 0
 c3327: - aux_cap_helper__b2,r13,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b2,r13,band_HIGH_PAIRED__condX1 <= 0
 c3328: - aux_cap_helper__b0,r10,band_UNPAIRED__condX2
        + aux_cap_helper__b0,r10,band_UNPAIRED__Z2 <= 0
 c3329: - aux_cap_helper__b8,r6,band_UNPAIRED__condX2
        + aux_cap_helper__b8,r6,band_UNPAIRED__Z2 <= 0
 c3330: aux_cap_helper__b7,r12,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b7,r12,band_HIGH_PAIRED__Z1 <= 0
 c3331: - aux_cap_helper__b0,r8,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b0,r8,band_HIGH_PAIRED__Z3 <= 0
 c3332: - sv_function_helpervar_b3,r13__condX1 <= 0
 c3333: aux_cap_helper__b6,r2,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b6,r2,band_HIGH_PAIRED__Z1 <= 0
 c3334: - X__b9,r0,band_HIGH_PAIRED
        + aux_cap_helper__b9,r0,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b9,r0,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b9,r0,band_HIGH_PAIRED__condX2  = 0
 c3335: sv_function_helpervar_b0,r9__Z2 + sv_function_helpervar_b0,r9__Z3
        + sv_function_helpervar_b0,r9__Z1  = 1
 c3336: - aux_cap_helper__b5,r11,band_UNPAIRED__condX1 <= 0
 c3337: - aux_c_b7,r8 + sv_function_helpervar_b7,r8__condX2
        + sv_function_helpervar_b7,r8__condX3
        + sv_function_helpervar_b7,r8__condX1  = 0
 c3338: 942.4308567147 sv_function_helpervar_b6,r12__condX1
        + 45.986304478 sv_function_helpervar_b6,r12__condX2
        + 14.7060062872 sv_function_helpervar_b6,r12__condX3 - aux_sv_b6,r12
        + 320.056859058612 sv_function_helpervar_b6,r12__Z3
        + 142.044406529987 sv_function_helpervar_b6,r12__Z2  = 0
 c3339: - 2 aux_cap_helper__b2,r13,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b2,r13,band_HIGH_PAIRED__condX2 <= 0
 c3340: - aux_cap_helper__b3,r7,band_UNPAIRED__condX2
        + aux_cap_helper__b3,r7,band_UNPAIRED__Z2 <= 0
 c3341: - aux_cap_helper__b4,r13,band_LOW_PAIRED__condX2
        + aux_cap_helper__b4,r13,band_LOW_PAIRED__Z2 <= 0
 c3342: aux_cap_helper__b7,r12,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b7,r12,band_HIGH_PAIRED__Z2 <= 0
 c3343: aux_cap_helper__b5,r2,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b5,r2,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b5,r2,band_HIGH_PAIRED__Z3  = 1
 c3344: aux_cap_helper__b3,r7,band_LOW_PAIRED__Z2
        - aux_cap_helper__b3,r7,band_LOW_PAIRED__condX2 <= 0
 c3345: aux_cap_helper__b0,r3,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b0,r3,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b0,r3,band_HIGH_PAIRED__Z2  = 1
 c3346: aux_cap_helper__b3,r12,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b3,r12,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b3,r12,band_HIGH_PAIRED__Z1  = 1
 c3347: - sv_function_helpervar_b8,r10__condX1 <= 0
 c3348: 0.158453086887582 sv_function_helpervar_b5,r12__Z2
        - sv_function_helpervar_b5,r12__condX2 <= 0
 c3349: aux_cap_helper__b6,r2,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b6,r2,band_HIGH_PAIRED__Z2 <= 0
 c3350: - 0.15845308693396 sv_function_helpervar_b7,r4__Z1
        + sv_function_helpervar_b7,r4__condX1 <= 0
 c3351: sv_function_helpervar_b2,r3__condX3
        + sv_function_helpervar_b2,r3__condX2
        + sv_function_helpervar_b2,r3__condX1 - aux_c_b2,r3  = 0
 c3352: aux_cap_helper__b2,r3,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b2,r3,band_HIGH_PAIRED__Z2 <= 0
 c3353: - aux_cap_helper__b2,r9,band_LOW_PAIRED__condX1 <= 0
 c3354: sv_function_helpervar_b6,r6__Z2 + sv_function_helpervar_b6,r6__Z1
        + sv_function_helpervar_b6,r6__Z3  = 1
 c3355: sv_function_helpervar_b6,r11__condX2
        - 5.65397694524761 sv_function_helpervar_b6,r11__Z2 <= 0
 c3356: - sv_function_helpervar_b2,r1__condX3
        + 6.0897224156964 sv_function_helpervar_b2,r1__Z3 <= 0
 c3357: aux_cap_helper__b2,r6,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b2,r6,band_HIGH_PAIRED__Z1 <= 0
 c3358: - 7.91255019037666 sv_function_helpervar_b7,r10__Z2
        + sv_function_helpervar_b7,r10__condX2 <= 0
 c3359: - aux_cap_helper__b5,r2,band_LOW_PAIRED__condX1 <= 0
 c3360: aux_cap_helper__b1,r4,band_UNPAIRED__Z2
        + aux_cap_helper__b1,r4,band_UNPAIRED__Z1  = 1
 c3361: - X__b5,r6,band_UNPAIRED + aux_cap_helper__b5,r6,band_UNPAIRED__condX1
        + aux_cap_helper__b5,r6,band_UNPAIRED__condX2  = 0
 c3362: - aux_cap_helper__b1,r3,band_LOW_PAIRED__condX1 <= 0
 c3363: - aux_c_b2,r6 + sv_function_helpervar_b2,r6__condX2
        + sv_function_helpervar_b2,r6__condX1
        + sv_function_helpervar_b2,r6__condX3  = 0
 c3364: - aux_cap_b9,r13,band_LOW_PAIRED
        - 1.2715511236 aux_cap_helper__b9,r13,band_LOW_PAIRED__Z2
        + 4.4504289328 aux_cap_helper__b9,r13,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b9,r13,band_LOW_PAIRED__condX1  = 0
 c3365: sv_function_helpervar_b0,r5__condX1
        - 0.158453086893356 sv_function_helpervar_b0,r5__Z1 <= 0
 c3366: sv_function_helpervar_b2,r11__Z3 + sv_function_helpervar_b2,r11__Z2
        + sv_function_helpervar_b2,r11__Z1  = 1
 c3367: sv_function_helpervar_b9,r3__condX2
        + sv_function_helpervar_b9,r3__condX1
        + sv_function_helpervar_b9,r3__condX3 - aux_c_b9,r3  = 0
 c3368: - sv_function_helpervar_b0,r5__condX1 <= 0
 c3369: 2 aux_cap_helper__b3,r8,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b3,r8,band_HIGH_PAIRED__condX3 <= 0
 c3370: aux_cap_helper__b5,r11,band_LOW_PAIRED__Z1
        + aux_cap_helper__b5,r11,band_LOW_PAIRED__Z2  = 1
 c3371: - aux_c_b7,r3 + aux_cap_b7,r3,band_UNPAIRED
        + aux_cap_b7,r3,band_LOW_PAIRED + aux_cap_b7,r3,band_HIGH_PAIRED  = 0
 c3372: - 0.158453086905199 sv_function_helpervar_b8,r3__Z1
        + sv_function_helpervar_b8,r3__condX1 <= 0
 c3373: aux_cap_helper__b2,r0,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b2,r0,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b2,r0,band_HIGH_PAIRED__condX2
        - X__b2,r0,band_HIGH_PAIRED  = 0
 c3374: aux_cap_helper__b9,r9,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b9,r9,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b9,r9,band_HIGH_PAIRED__condX1
        - X__b9,r9,band_HIGH_PAIRED  = 0
 c3375: sv_function_helpervar_b9,r2__condX2
        + sv_function_helpervar_b9,r2__condX3
        + sv_function_helpervar_b9,r2__condX1 - aux_c_b9,r2  = 0
 c3376: aux_cap_helper__b6,r5,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b6,r5,band_HIGH_PAIRED__condX2 <= 0
 c3377: - X__b6,r3,band_LOW_PAIRED
        + aux_cap_helper__b6,r3,band_LOW_PAIRED__condX1
        + aux_cap_helper__b6,r3,band_LOW_PAIRED__condX2  = 0
 c3378: sv_function_helpervar_b5,r8__condX2
        - 5.7076421829379 sv_function_helpervar_b5,r8__Z2 <= 0
 c3379: aux_cap_helper__b0,r2,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b0,r2,band_HIGH_PAIRED__Z3 <= 0
 c3380: - 2 aux_cap_helper__b9,r2,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b9,r2,band_HIGH_PAIRED__condX2 <= 0
 c3381: - aux_cap_b4,r12,band_LOW_PAIRED
        - 1.2715511236 aux_cap_helper__b4,r12,band_LOW_PAIRED__Z2
        + 4.4504289328 aux_cap_helper__b4,r12,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b4,r12,band_LOW_PAIRED__condX1  = 0
 c3382: aux_cap_helper__b7,r4,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b7,r4,band_HIGH_PAIRED__condX2 <= 0
 c3383: - X__b9,r11,band_LOW_PAIRED
        + aux_cap_helper__b9,r11,band_LOW_PAIRED__condX1
        + aux_cap_helper__b9,r11,band_LOW_PAIRED__condX2  = 0
 c3384: - X__b3,r0,band_UNPAIRED + aux_cap_helper__b3,r0,band_UNPAIRED__condX1
        + aux_cap_helper__b3,r0,band_UNPAIRED__condX2  = 0
 c3385: 5.87988364123941 sv_function_helpervar_b5,r12__Z3
        - sv_function_helpervar_b5,r12__condX3 <= 0
 c3386: 2 aux_cap_helper__b6,r5,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b6,r5,band_HIGH_PAIRED__condX3 <= 0
 c3387: aux_cap_helper__b4,r0,band_LOW_PAIRED__Z2
        - aux_cap_helper__b4,r0,band_LOW_PAIRED__condX2 <= 0
 c3388: 2 aux_cap_helper__b7,r4,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b7,r4,band_HIGH_PAIRED__condX3 <= 0
 c3389: - sv_function_helpervar_b2,r0__condX3
        + 5.90029274601902 sv_function_helpervar_b2,r0__Z3 <= 0
 c3390: - aux_cap_helper__b8,r9,band_HIGH_PAIRED__condX1 <= 0
 c3391: - 0.253841888 aux_cap_helper__b9,r3,band_UNPAIRED__Z2
        - aux_cap_b9,r3,band_UNPAIRED
        + 0.888446608 aux_cap_helper__b9,r3,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b9,r3,band_UNPAIRED__condX1  = 0
 c3392: sv_function_helpervar_b3,r3__condX3
        - 15.8453086887446 sv_function_helpervar_b3,r3__Z3 <= 0
 c3393: - aux_c_b2,r7 + sv_function_helpervar_b2,r7__condX3
        + sv_function_helpervar_b2,r7__condX2
        + sv_function_helpervar_b2,r7__condX1  = 0
 c3394: aux_cap_helper__b4,r10,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b4,r10,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b4,r10,band_HIGH_PAIRED__Z2  = 1
 c3395: - aux_cap_helper__b3,r5,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b3,r5,band_HIGH_PAIRED__Z3 <= 0
 c3396: aux_cap_helper__b0,r5,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b0,r5,band_HIGH_PAIRED__Z3 <= 0
 c3397: - sv_function_helpervar_b0,r2__condX2
        + 0.158453086904217 sv_function_helpervar_b0,r2__Z2 <= 0
 c3398: 5.78636564086377 sv_function_helpervar_b5,r2__Z3
        - sv_function_helpervar_b5,r2__condX3 <= 0
 c3399: - aux_cap_helper__b2,r5,band_LOW_PAIRED__condX1 <= 0
 c3400: - sv_function_helpervar_b5,r6__condX1 <= 0
 c3401: aux_cap_helper__b3,r8,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b3,r8,band_HIGH_PAIRED__condX2 <= 0
 c3402: aux_cap_helper__b5,r6,band_LOW_PAIRED__condX1
        - aux_cap_helper__b5,r6,band_LOW_PAIRED__Z1 <= 0
 c3403: aux_cap_helper__b6,r7,band_UNPAIRED__Z2
        - aux_cap_helper__b6,r7,band_UNPAIRED__condX2 <= 0
 c3404: 191.109263989905 sv_function_helpervar_b7,r8__Z2
        + 380.645869044714 sv_function_helpervar_b7,r8__Z3 - aux_sv_b7,r8
        + 49.5757770809 sv_function_helpervar_b7,r8__condX2
        + 22.4836530609 sv_function_helpervar_b7,r8__condX3
        + 1255.6694397299 sv_function_helpervar_b7,r8__condX1  = 0
 c3405: - X__b2,r7,band_LOW_PAIRED
        + aux_cap_helper__b2,r7,band_LOW_PAIRED__condX2
        + aux_cap_helper__b2,r7,band_LOW_PAIRED__condX1  = 0
 c3406: aux_cap_helper__b6,r0,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b6,r0,band_HIGH_PAIRED__Z3 <= 0
 c3407: aux_cap_helper__b5,r6,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b5,r6,band_LOW_PAIRED__Z2 <= 0
 c3408: aux_cap_helper__b0,r6,band_LOW_PAIRED__condX1
        - aux_cap_helper__b0,r6,band_LOW_PAIRED__Z1 <= 0
 c3409: - aux_cap_helper__b6,r1,band_UNPAIRED__condX1 <= 0
 c3410: aux_cap_helper__b6,r5,band_LOW_PAIRED__condX1
        - aux_cap_helper__b6,r5,band_LOW_PAIRED__Z1 <= 0
 c3411: aux_cap_helper__b5,r9,band_UNPAIRED__condX1
        - aux_cap_helper__b5,r9,band_UNPAIRED__Z1 <= 0
 c3412: sv_function_helpervar_b8,r0__condX1
        - 0.158453086899447 sv_function_helpervar_b8,r0__Z1 <= 0
 c3413: 399.141874056224 sv_function_helpervar_b3,r0__Z3
        + 173.170505219692 sv_function_helpervar_b3,r0__Z2
        + 1152.1877366202 sv_function_helpervar_b3,r0__condX1
        + 59.305871042 sv_function_helpervar_b3,r0__condX2
        + 17.4837115029 sv_function_helpervar_b3,r0__condX3 - aux_sv_b3,r0  = 0
 c3414: - aux_cap_helper__b5,r3,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b5,r3,band_HIGH_PAIRED__condX1 <= 0
 c3415: aux_cap_helper__b9,r6,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b9,r6,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b9,r6,band_HIGH_PAIRED__condX2
        - X__b9,r6,band_HIGH_PAIRED  = 0
 c3416: sv_function_helpervar_b3,r11__condX1
        + sv_function_helpervar_b3,r11__condX2
        + sv_function_helpervar_b3,r11__condX3 - aux_c_b3,r11  = 0
 c3417: aux_cap_helper__b0,r6,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b0,r6,band_LOW_PAIRED__Z2 <= 0
 c3418: - aux_c_b5,r5 + sv_function_helpervar_b5,r5__condX3
        + sv_function_helpervar_b5,r5__condX2
        + sv_function_helpervar_b5,r5__condX1  = 0
 c3419: aux_cap_helper__b9,r9,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b9,r9,band_HIGH_PAIRED__Z1 <= 0
 c3420: - aux_cap_helper__b2,r7,band_UNPAIRED__Z1
        + aux_cap_helper__b2,r7,band_UNPAIRED__condX1 <= 0
 c3421: aux_cap_helper__b5,r9,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b5,r9,band_UNPAIRED__Z2 <= 0
 c3422: 34.4685079234 sv_function_helpervar_b9,r13__condX3
        + 82.6848168355 sv_function_helpervar_b9,r13__condX2
        + 1995.2925263063 sv_function_helpervar_b9,r13__condX1
        + 303.058595718839 sv_function_helpervar_b9,r13__Z2
        + 624.799778549948 sv_function_helpervar_b9,r13__Z3 - aux_sv_b9,r13
         = 0
 c3423: - 2 aux_cap_helper__b5,r3,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b5,r3,band_HIGH_PAIRED__condX2 <= 0
 c3424: aux_cap_helper__b3,r8,band_LOW_PAIRED__Z1
        + aux_cap_helper__b3,r8,band_LOW_PAIRED__Z2  = 1
 c3425: - X__b7,r6,band_LOW_PAIRED
        + aux_cap_helper__b7,r6,band_LOW_PAIRED__condX1
        + aux_cap_helper__b7,r6,band_LOW_PAIRED__condX2  = 0
 c3426: aux_cap_helper__b5,r10,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b5,r10,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b5,r10,band_HIGH_PAIRED__Z2  = 1
 c3427: - aux_sv_b0,r9 + 24.6517414920114 sv_function_helpervar_b0,r9__Z2
        + 56.0759535642533 sv_function_helpervar_b0,r9__Z3
        + 163.7425743555 sv_function_helpervar_b0,r9__condX1
        + 2.5255772731 sv_function_helpervar_b0,r9__condX3
        + 8.1650342238 sv_function_helpervar_b0,r9__condX2  = 0
 c3428: aux_cap_helper__b6,r10,band_UNPAIRED__Z1
        + aux_cap_helper__b6,r10,band_UNPAIRED__Z2  = 1
 c3429: X__b9,r5,band_UNPAIRED + X__b9,r5,band_HIGH_PAIRED
        + X__b9,r5,band_LOW_PAIRED - GlobalBidder_W_ir__b9,r5 >= 0
 c3430: aux_cap_helper__b9,r3,band_UNPAIRED__Z2
        - aux_cap_helper__b9,r3,band_UNPAIRED__condX2 <= 0
 c3431: - aux_cap_b5,r1,band_HIGH_PAIRED
        - 0.8112667416 aux_cap_helper__b5,r1,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b5,r1,band_HIGH_PAIRED__Z3
        + 2.0281668543 aux_cap_helper__b5,r1,band_HIGH_PAIRED__condX1
        + 2.8394335959 aux_cap_helper__b5,r1,band_HIGH_PAIRED__condX2
        + 1.8253501688 aux_cap_helper__b5,r1,band_HIGH_PAIRED__condX3  = 0
 c3432: aux_cap_helper__b1,r13,band_LOW_PAIRED__Z2
        - aux_cap_helper__b1,r13,band_LOW_PAIRED__condX2 <= 0
 c3433: - 6.25459312389671 sv_function_helpervar_b5,r7__Z2
        + sv_function_helpervar_b5,r7__condX2 <= 0
 c3434: - aux_sv_b2,r1 + 114.7166939308 sv_function_helpervar_b2,r1__condX1
        + 1.8632621111 sv_function_helpervar_b2,r1__condX3
        + 5.2212411147 sv_function_helpervar_b2,r1__condX2
        + 17.3498925018197 sv_function_helpervar_b2,r1__Z2
        + 37.799052511471 sv_function_helpervar_b2,r1__Z3  = 0
 c3435: aux_cap_helper__b0,r7,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b0,r7,band_HIGH_PAIRED__Z3 <= 0
 c3436: - 2 aux_cap_helper__b0,r1,band_UNPAIRED__Z2
        + aux_cap_helper__b0,r1,band_UNPAIRED__condX2 <= 0
 c3437: sv_function_helpervar_b7,r13__condX2
        - 7.48392491398104 sv_function_helpervar_b7,r13__Z2 <= 0
 c3438: aux_cap_helper__b7,r0,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b7,r0,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b7,r0,band_HIGH_PAIRED__Z1  = 1
 c3439: sv_function_helpervar_b8,r7__Z1 + sv_function_helpervar_b8,r7__Z3
        + sv_function_helpervar_b8,r7__Z2  = 1
 c3440: - aux_cap_helper__b6,r11,band_UNPAIRED__condX2
        + aux_cap_helper__b6,r11,band_UNPAIRED__Z2 <= 0
 c3441: sv_function_helpervar_b4,r13__condX3
        + sv_function_helpervar_b4,r13__condX1
        + sv_function_helpervar_b4,r13__condX2 - aux_c_b4,r13  = 0
 c3442: aux_cap_helper__b8,r4,band_LOW_PAIRED__Z2
        + aux_cap_helper__b8,r4,band_LOW_PAIRED__Z1  = 1
 c3443: - aux_cap_helper__b3,r10,band_HIGH_PAIRED__condX1 <= 0
 c3444: aux_cap_helper__b7,r9,band_LOW_PAIRED__Z2
        - aux_cap_helper__b7,r9,band_LOW_PAIRED__condX2 <= 0
 c3445: - aux_sv_b0,r7 + 24.7570340663905 sv_function_helpervar_b0,r7__Z2
        + 54.1160808073709 sv_function_helpervar_b0,r7__Z3
        + 163.7425743633 sv_function_helpervar_b0,r7__condX1
        + 7.5005310805 sv_function_helpervar_b0,r7__condX2
        + 2.6492651605 sv_function_helpervar_b0,r7__condX3  = 0
 c3446: aux_cap_helper__b7,r3,band_UNPAIRED__condX1
        + aux_cap_helper__b7,r3,band_UNPAIRED__condX2 - X__b7,r3,band_UNPAIRED
         = 0
 c3447: X__b7,r0,band_UNPAIRED - GlobalBidder_W_ir__b7,r0
        + X__b7,r0,band_HIGH_PAIRED + X__b7,r0,band_LOW_PAIRED >= 0
 c3448: - aux_cap_b6,r12,band_LOW_PAIRED
        - 1.2715511236 aux_cap_helper__b6,r12,band_LOW_PAIRED__Z2
        + 4.4504289328 aux_cap_helper__b6,r12,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b6,r12,band_LOW_PAIRED__condX1  = 0
 c3449: 6.65803747744876 sv_function_helpervar_b7,r1__Z3
        - sv_function_helpervar_b7,r1__condX3 <= 0
 c3450: - sv_function_helpervar_b2,r3__condX1 <= 0
 c3451: aux_cap_helper__b5,r12,band_UNPAIRED__Z2
        + aux_cap_helper__b5,r12,band_UNPAIRED__Z1  = 1
 c3452: - aux_cap_helper__b3,r1,band_LOW_PAIRED__condX2
        + aux_cap_helper__b3,r1,band_LOW_PAIRED__Z2 <= 0
 c3453: aux_cap_helper__b3,r1,band_UNPAIRED__Z2
        + aux_cap_helper__b3,r1,band_UNPAIRED__Z1  = 1
 c3454: sv_function_helpervar_b5,r9__condX3
        - 15.8453086887446 sv_function_helpervar_b5,r9__Z3 <= 0
 c3455: - X__b5,r8,band_LOW_PAIRED
        + aux_cap_helper__b5,r8,band_LOW_PAIRED__condX1
        + aux_cap_helper__b5,r8,band_LOW_PAIRED__condX2  = 0
 c3456: - aux_cap_helper__b9,r3,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b9,r3,band_HIGH_PAIRED__Z3 <= 0
 c3457: - aux_Omega_b1,r9 + 255.355377243079 aux_sv_b1,r9  = 0
 c3458: 4.4504289328 aux_cap_helper__b7,r13,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b7,r13,band_LOW_PAIRED__condX1
        - aux_cap_b7,r13,band_LOW_PAIRED
        - 1.2715511236 aux_cap_helper__b7,r13,band_LOW_PAIRED__Z2  = 0
 c3459: aux_cap_helper__b6,r5,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b6,r5,band_LOW_PAIRED__Z2 <= 0
 c3460: 0.158453086953743 sv_function_helpervar_b9,r9__Z2
        - sv_function_helpervar_b9,r9__condX2 <= 0
 c3461: aux_cap_helper__b2,r5,band_UNPAIRED__Z2
        + aux_cap_helper__b2,r5,band_UNPAIRED__Z1  = 1
 c3462: 1.21690011250031 aux_cap_helper__b5,r13,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b5,r13,band_HIGH_PAIRED__Z2
        + 1.8253501688 aux_cap_helper__b5,r13,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b5,r13,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b5,r13,band_HIGH_PAIRED__condX1
        - aux_cap_b5,r13,band_HIGH_PAIRED  = 0
 c3463: - aux_cap_helper__b4,r6,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b4,r6,band_HIGH_PAIRED__Z2 <= 0
 c3464: 0.888446608 aux_cap_helper__b6,r10,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b6,r10,band_UNPAIRED__condX1
        - 0.253841888 aux_cap_helper__b6,r10,band_UNPAIRED__Z2
        - aux_cap_b6,r10,band_UNPAIRED  = 0
 c3465: sv_function_helpervar_b8,r9__Z3 + sv_function_helpervar_b8,r9__Z2
        + sv_function_helpervar_b8,r9__Z1  = 1
 c3466: - aux_cap_helper__b0,r1,band_UNPAIRED__Z1
        + aux_cap_helper__b0,r1,band_UNPAIRED__condX1 <= 0
 c3467: aux_cap_helper__b4,r4,band_UNPAIRED__condX2
        + aux_cap_helper__b4,r4,band_UNPAIRED__condX1 - X__b4,r4,band_UNPAIRED
         = 0
 c3468: aux_cap_b1,r6,band_UNPAIRED + aux_cap_b1,r6,band_HIGH_PAIRED
        - aux_c_b1,r6 + aux_cap_b1,r6,band_LOW_PAIRED  = 0
 c3469: - X__b7,r2,band_HIGH_PAIRED
        + aux_cap_helper__b7,r2,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b7,r2,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b7,r2,band_HIGH_PAIRED__condX1  = 0
 c3470: aux_cap_helper__b9,r3,band_UNPAIRED__Z1
        + aux_cap_helper__b9,r3,band_UNPAIRED__Z2  = 1
 c3471: aux_cap_helper__b0,r12,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b0,r12,band_LOW_PAIRED__Z2 <= 0
 c3472: sv_function_helpervar_b3,r9__condX2
        - 5.42311341710275 sv_function_helpervar_b3,r9__Z2 <= 0
 c3473: - 2 aux_cap_helper__b2,r7,band_UNPAIRED__Z2
        + aux_cap_helper__b2,r7,band_UNPAIRED__condX2 <= 0
 c3474: - 2 aux_cap_helper__b3,r4,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b3,r4,band_HIGH_PAIRED__condX2 <= 0
 c3475: 0.888446608 aux_cap_helper__b5,r1,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b5,r1,band_UNPAIRED__condX1
        - aux_cap_b5,r1,band_UNPAIRED
        - 0.253841888 aux_cap_helper__b5,r1,band_UNPAIRED__Z2  = 0
 c3476: X__b7,r11,band_UNPAIRED + X__b2,r11,band_UNPAIRED
        + X__b9,r11,band_UNPAIRED + X__b1,r11,band_UNPAIRED
        + X__b5,r11,band_UNPAIRED + X__b3,r11,band_UNPAIRED
        + X__b6,r11,band_UNPAIRED + X__b0,r11,band_UNPAIRED
        + X__b4,r11,band_UNPAIRED + X__b8,r11,band_UNPAIRED <= 2
 c3477: aux_cap_helper__b3,r1,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b3,r1,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b3,r1,band_HIGH_PAIRED__Z2  = 1
 c3478: 0.888446608 aux_cap_helper__b8,r3,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b8,r3,band_UNPAIRED__condX1
        - 0.253841888 aux_cap_helper__b8,r3,band_UNPAIRED__Z2
        - aux_cap_b8,r3,band_UNPAIRED  = 0
 c3479: aux_cap_helper__b9,r9,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b9,r9,band_HIGH_PAIRED__Z2 <= 0
 c3480: - sv_function_helpervar_b1,r7__condX3
        + 5.60179427758248 sv_function_helpervar_b1,r7__Z3 <= 0
 c3481: aux_cap_helper__b1,r13,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b1,r13,band_UNPAIRED__Z2 <= 0
 c3482: aux_cap_helper__b8,r7,band_UNPAIRED__Z1
        + aux_cap_helper__b8,r7,band_UNPAIRED__Z2  = 1
 c3483: aux_cap_helper__b4,r0,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b4,r0,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b4,r0,band_HIGH_PAIRED__Z1  = 1
 c3484: - 0.253841888 aux_cap_helper__b4,r10,band_UNPAIRED__Z2
        - aux_cap_b4,r10,band_UNPAIRED
        + 0.888446608 aux_cap_helper__b4,r10,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b4,r10,band_UNPAIRED__condX1  = 0
 c3485: 0.0102040816326531 X__b8,r1,band_LOW_PAIRED
        + 0.0102040816326531 X__b8,r1,band_HIGH_PAIRED
        + 0.0102040816326531 X__b8,r1,band_UNPAIRED - GlobalBidder_W_ir__b8,r1
        <= 0
 c3486: - X__b3,r11,band_UNPAIRED
        + aux_cap_helper__b3,r11,band_UNPAIRED__condX1
        + aux_cap_helper__b3,r11,band_UNPAIRED__condX2  = 0
 c3487: - aux_Omega_b1,r8 + 243.926751606787 aux_sv_b1,r8  = 0
 c3488: aux_cap_helper__b9,r9,band_UNPAIRED__condX1
        + aux_cap_helper__b9,r9,band_UNPAIRED__condX2 - X__b9,r9,band_UNPAIRED
         = 0
 c3489: aux_cap_helper__b7,r6,band_UNPAIRED__Z1
        + aux_cap_helper__b7,r6,band_UNPAIRED__Z2  = 1
 c3490: - 2 aux_cap_helper__b3,r9,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b3,r9,band_HIGH_PAIRED__condX2 <= 0
 c3491: aux_cap_helper__b8,r2,band_UNPAIRED__condX2
        + aux_cap_helper__b8,r2,band_UNPAIRED__condX1 - X__b8,r2,band_UNPAIRED
         = 0
 c3492: - aux_cap_helper__b2,r2,band_LOW_PAIRED__condX1 <= 0
 c3493: - aux_cap_helper__b1,r0,band_LOW_PAIRED__condX1 <= 0
 c3494: - aux_cap_b2,r10,band_HIGH_PAIRED
        + 1.8253501688 aux_cap_helper__b2,r10,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b2,r10,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b2,r10,band_HIGH_PAIRED__condX1
        - 0.8112667416 aux_cap_helper__b2,r10,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b2,r10,band_HIGH_PAIRED__Z3  = 0
 c3495: - sv_function_helpervar_b1,r12__condX1 <= 0
 c3496: - 3 aux_cap_helper__b6,r7,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b6,r7,band_HIGH_PAIRED__condX3 <= 0
 c3497: aux_cap_b2,r4,band_HIGH_PAIRED - aux_c_b2,r4
        + aux_cap_b2,r4,band_LOW_PAIRED + aux_cap_b2,r4,band_UNPAIRED  = 0
 c3498: - 3 aux_cap_helper__b0,r0,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b0,r0,band_HIGH_PAIRED__condX3 <= 0
 c3499: aux_cap_helper__b0,r12,band_LOW_PAIRED__condX1
        - aux_cap_helper__b0,r12,band_LOW_PAIRED__Z1 <= 0
 c3500: - aux_cap_helper__b6,r7,band_LOW_PAIRED__condX1 <= 0
 c3501: - aux_cap_b6,r10,band_LOW_PAIRED
        - 1.2715511236 aux_cap_helper__b6,r10,band_LOW_PAIRED__Z2
        + 3.1788778092 aux_cap_helper__b6,r10,band_LOW_PAIRED__condX1
        + 4.4504289328 aux_cap_helper__b6,r10,band_LOW_PAIRED__condX2  = 0
 c3502: sv_function_helpervar_b1,r4__Z3 + sv_function_helpervar_b1,r4__Z2
        + sv_function_helpervar_b1,r4__Z1  = 1
 c3503: aux_cap_helper__b2,r1,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b2,r1,band_HIGH_PAIRED__Z1 <= 0
 c3504: 28.5247999286551 aux_sv_b4,r13 - aux_Omega_b4,r13  = 0
 c3505: - X__b5,r1,band_LOW_PAIRED
        + aux_cap_helper__b5,r1,band_LOW_PAIRED__condX2
        + aux_cap_helper__b5,r1,band_LOW_PAIRED__condX1  = 0
 c3506: aux_cap_helper__b1,r13,band_UNPAIRED__condX1
        - aux_cap_helper__b1,r13,band_UNPAIRED__Z1 <= 0
 c3507: aux_cap_helper__b6,r1,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b6,r1,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b6,r1,band_HIGH_PAIRED__Z3  = 1
 c3508: - aux_cap_helper__b6,r0,band_LOW_PAIRED__condX2
        + aux_cap_helper__b6,r0,band_LOW_PAIRED__Z2 <= 0
 c3509: aux_cap_b3,r2,band_HIGH_PAIRED - aux_c_b3,r2
        + aux_cap_b3,r2,band_UNPAIRED + aux_cap_b3,r2,band_LOW_PAIRED  = 0
 c3510: - aux_cap_helper__b4,r6,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b4,r6,band_HIGH_PAIRED__Z3 <= 0
 c3511: 2 aux_cap_helper__b3,r3,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b3,r3,band_HIGH_PAIRED__condX3 <= 0
 c3512: aux_cap_helper__b0,r0,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b0,r0,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b0,r0,band_HIGH_PAIRED__Z3  = 1
 c3513: sv_function_helpervar_b8,r2__condX3
        - 15.8453086887446 sv_function_helpervar_b8,r2__Z3 <= 0
 c3514: aux_cap_helper__b1,r8,band_UNPAIRED__Z1
        + aux_cap_helper__b1,r8,band_UNPAIRED__Z2  = 1
 c3515: - aux_cap_b1,r2,band_HIGH_PAIRED
        + 1.8253501688 aux_cap_helper__b1,r2,band_HIGH_PAIRED__condX3
        + 2.0281668543 aux_cap_helper__b1,r2,band_HIGH_PAIRED__condX1
        + 2.8394335959 aux_cap_helper__b1,r2,band_HIGH_PAIRED__condX2
        - 0.8112667416 aux_cap_helper__b1,r2,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b1,r2,band_HIGH_PAIRED__Z3  = 0
 c3516: aux_cap_helper__b8,r7,band_UNPAIRED__Z2
        - aux_cap_helper__b8,r7,band_UNPAIRED__condX2 <= 0
 c3517: - aux_cap_helper__b3,r9,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b3,r9,band_HIGH_PAIRED__condX1 <= 0
 c3518: - aux_cap_b0,r11,band_HIGH_PAIRED
        + 1.8253501688 aux_cap_helper__b0,r11,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b0,r11,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b0,r11,band_HIGH_PAIRED__condX1
        - 0.8112667416 aux_cap_helper__b0,r11,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b0,r11,band_HIGH_PAIRED__Z3  = 0
 c3519: aux_cap_b6,r2,band_UNPAIRED - aux_c_b6,r2
        + aux_cap_b6,r2,band_LOW_PAIRED + aux_cap_b6,r2,band_HIGH_PAIRED  = 0
 c3520: sv_function_helpervar_b8,r13__condX1
        + sv_function_helpervar_b8,r13__condX2
        + sv_function_helpervar_b8,r13__condX3 - aux_c_b8,r13  = 0
 c3521: aux_cap_b7,r0,band_UNPAIRED - aux_c_b7,r0
        + aux_cap_b7,r0,band_LOW_PAIRED + aux_cap_b7,r0,band_HIGH_PAIRED  = 0
 c3522: - sv_function_helpervar_b3,r12__condX1 <= 0
 c3523: - aux_Omega_b4,r3 + 232.562076775097 aux_sv_b4,r3  = 0
 c3524: - aux_cap_helper__b2,r7,band_HIGH_PAIRED__condX1 <= 0
 c3525: - X__b4,r4,band_HIGH_PAIRED
        + aux_cap_helper__b4,r4,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b4,r4,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b4,r4,band_HIGH_PAIRED__condX3  = 0
 c3526: - aux_cap_helper__b9,r3,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b9,r3,band_HIGH_PAIRED__Z2 <= 0
 c3527: 0.158453086936519 sv_function_helpervar_b9,r1__Z2
        - sv_function_helpervar_b9,r1__condX2 <= 0
 c3528: X__b3,r0,band_HIGH_PAIRED + X__b4,r0,band_HIGH_PAIRED
        + X__b6,r0,band_HIGH_PAIRED + X__b9,r0,band_HIGH_PAIRED
        + X__b8,r0,band_HIGH_PAIRED + X__b7,r0,band_HIGH_PAIRED
        + X__b1,r0,band_HIGH_PAIRED + X__b5,r0,band_HIGH_PAIRED
        + X__b2,r0,band_HIGH_PAIRED + X__b0,r0,band_HIGH_PAIRED <= 3
 c3529: aux_cap_helper__b4,r6,band_LOW_PAIRED__Z2
        + aux_cap_helper__b4,r6,band_LOW_PAIRED__Z1  = 1
 c3530: sv_function_helpervar_b4,r8__condX3
        - 15.8453086887446 sv_function_helpervar_b4,r8__Z3 <= 0
 c3531: 942.4308564949 sv_function_helpervar_b6,r13__condX1
        + 44.3614970045 sv_function_helpervar_b6,r13__condX2
        + 15.0054428552 sv_function_helpervar_b6,r13__condX3
        + 315.312194206097 sv_function_helpervar_b6,r13__Z3
        + 142.301862288125 sv_function_helpervar_b6,r13__Z2 - aux_sv_b6,r13
         = 0
 c3532: aux_cap_helper__b9,r13,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b9,r13,band_HIGH_PAIRED__condX2 <= 0
 c3533: aux_cap_helper__b7,r10,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b7,r10,band_HIGH_PAIRED__Z1 <= 0
 c3534: - sv_function_helpervar_b6,r6__condX1 <= 0
 c3535: 0.158453086934665 sv_function_helpervar_b3,r5__Z2
        - sv_function_helpervar_b3,r5__condX2 <= 0
 c3536: aux_cap_b4,r0,band_HIGH_PAIRED + aux_cap_b4,r0,band_UNPAIRED
        + aux_cap_b4,r0,band_LOW_PAIRED - aux_c_b4,r0  = 0
 c3537: - 6.47958443516442 sv_function_helpervar_b1,r0__Z2
        + sv_function_helpervar_b1,r0__condX2 <= 0
 c3538: - aux_cap_helper__b7,r10,band_UNPAIRED__condX1 <= 0
 c3539: 5.55639826435479 sv_function_helpervar_b0,r1__Z3
        - sv_function_helpervar_b0,r1__condX3 <= 0
 c3540: 0.63460472 aux_cap_helper__b3,r7,band_UNPAIRED__condX1
        + 0.888446608 aux_cap_helper__b3,r7,band_UNPAIRED__condX2
        - 0.253841888 aux_cap_helper__b3,r7,band_UNPAIRED__Z2
        - aux_cap_b3,r7,band_UNPAIRED  = 0
 c3541: 2 aux_cap_helper__b9,r13,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b9,r13,band_HIGH_PAIRED__condX3 <= 0
 c3542: aux_cap_helper__b3,r10,band_LOW_PAIRED__Z2
        + aux_cap_helper__b3,r10,band_LOW_PAIRED__Z1  = 1
 c3543: sv_function_helpervar_b3,r8__condX1
        - 0.15845308695304 sv_function_helpervar_b3,r8__Z1 <= 0
 c3544: aux_cap_helper__b7,r10,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b7,r10,band_HIGH_PAIRED__Z2 <= 0
 c3545: - sv_function_helpervar_b7,r3__condX1 <= 0
 c3546: - aux_cap_helper__b1,r7,band_UNPAIRED__condX2
        + aux_cap_helper__b1,r7,band_UNPAIRED__Z2 <= 0
 c3547: aux_cap_helper__b3,r3,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b3,r3,band_HIGH_PAIRED__condX2 <= 0
 c3548: - aux_c_b6,r4 + sv_function_helpervar_b6,r4__condX1
        + sv_function_helpervar_b6,r4__condX2
        + sv_function_helpervar_b6,r4__condX3  = 0
 c3549: - 15.8453086887446 sv_function_helpervar_b8,r13__Z3
        + sv_function_helpervar_b8,r13__condX3 <= 0
 c3550: aux_cap_helper__b9,r1,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b9,r1,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b9,r1,band_HIGH_PAIRED__Z2  = 1
 c3551: - aux_cap_helper__b2,r3,band_UNPAIRED__condX1 <= 0
 c3552: - aux_c_b7,r8 + aux_cap_b7,r8,band_UNPAIRED
        + aux_cap_b7,r8,band_LOW_PAIRED + aux_cap_b7,r8,band_HIGH_PAIRED  = 0
 c3553: 3.1788778092 aux_cap_helper__b1,r9,band_LOW_PAIRED__condX1
        + 4.4504289328 aux_cap_helper__b1,r9,band_LOW_PAIRED__condX2
        - aux_cap_b1,r9,band_LOW_PAIRED
        - 1.2715511236 aux_cap_helper__b1,r9,band_LOW_PAIRED__Z2  = 0
 c3554: - 15.8453086887446 sv_function_helpervar_b0,r9__Z3
        + sv_function_helpervar_b0,r9__condX3 <= 0
 c3555: - aux_cap_helper__b7,r2,band_LOW_PAIRED__condX2
        + aux_cap_helper__b7,r2,band_LOW_PAIRED__Z2 <= 0
 c3556: aux_cap_helper__b0,r5,band_UNPAIRED__Z2
        - aux_cap_helper__b0,r5,band_UNPAIRED__condX2 <= 0
 c3557: sv_function_helpervar_b9,r6__condX2
        - 7.9811679679664 sv_function_helpervar_b9,r6__Z2 <= 0
 c3558: aux_cap_helper__b2,r1,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b2,r1,band_HIGH_PAIRED__Z2 <= 0
 c3559: - X__b0,r6,band_LOW_PAIRED
        + aux_cap_helper__b0,r6,band_LOW_PAIRED__condX2
        + aux_cap_helper__b0,r6,band_LOW_PAIRED__condX1  = 0
 c3560: sv_function_helpervar_b4,r13__condX2
        - 6.26560685803533 sv_function_helpervar_b4,r13__Z2 <= 0
 c3561: - GlobalBidder_W_i_8 + GlobalBidder_W_ir__b8,r10
        + GlobalBidder_W_ir__b8,r11 + GlobalBidder_W_ir__b8,r12
        + GlobalBidder_W_ir__b8,r13 + GlobalBidder_W_ir__b8,r8
        + GlobalBidder_W_ir__b8,r7 + GlobalBidder_W_ir__b8,r6
        + GlobalBidder_W_ir__b8,r5 + GlobalBidder_W_ir__b8,r9
        + GlobalBidder_W_ir__b8,r0 + GlobalBidder_W_ir__b8,r4
        + GlobalBidder_W_ir__b8,r3 + GlobalBidder_W_ir__b8,r2
        + GlobalBidder_W_ir__b8,r1  = 0
 c3562: - aux_cap_helper__b1,r1,band_UNPAIRED__condX1 <= 0
 c3563: aux_cap_helper__b5,r8,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b5,r8,band_UNPAIRED__Z2 <= 0
 c3564: - aux_c_b0,r0 + sv_function_helpervar_b0,r0__condX3
        + sv_function_helpervar_b0,r0__condX2
        + sv_function_helpervar_b0,r0__condX1  = 0
 c3565: aux_cap_helper__b5,r3,band_UNPAIRED__condX2
        + aux_cap_helper__b5,r3,band_UNPAIRED__condX1 - X__b5,r3,band_UNPAIRED
         = 0
 c3566: - 1.2715511236 aux_cap_helper__b6,r4,band_LOW_PAIRED__Z2
        - aux_cap_b6,r4,band_LOW_PAIRED
        + 3.1788778092 aux_cap_helper__b6,r4,band_LOW_PAIRED__condX1
        + 4.4504289328 aux_cap_helper__b6,r4,band_LOW_PAIRED__condX2  = 0
 c3567: aux_cap_helper__b2,r0,band_LOW_PAIRED__condX2
        + aux_cap_helper__b2,r0,band_LOW_PAIRED__condX1
        - X__b2,r0,band_LOW_PAIRED  = 0
 c3568: aux_cap_helper__b1,r10,band_UNPAIRED__Z2
        - aux_cap_helper__b1,r10,band_UNPAIRED__condX2 <= 0
 c3569: - aux_cap_b8,r6,band_LOW_PAIRED
        + 4.4504289328 aux_cap_helper__b8,r6,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b8,r6,band_LOW_PAIRED__condX1
        - 1.2715511236 aux_cap_helper__b8,r6,band_LOW_PAIRED__Z2  = 0
 c3570: X__b7,r11,band_LOW_PAIRED + X__b0,r11,band_LOW_PAIRED
        + X__b3,r11,band_LOW_PAIRED + X__b6,r11,band_LOW_PAIRED
        + X__b4,r11,band_LOW_PAIRED + X__b8,r11,band_LOW_PAIRED
        + X__b5,r11,band_LOW_PAIRED + X__b2,r11,band_LOW_PAIRED
        + X__b1,r11,band_LOW_PAIRED + X__b9,r11,band_LOW_PAIRED <= 2
 c3571: - X__b4,r1,band_HIGH_PAIRED
        + aux_cap_helper__b4,r1,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b4,r1,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b4,r1,band_HIGH_PAIRED__condX3  = 0
 c3572: sv_function_helpervar_b8,r12__Z2 + sv_function_helpervar_b8,r12__Z3
        + sv_function_helpervar_b8,r12__Z1  = 1
 c3573: 1.21690011250031 aux_cap_helper__b9,r0,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b9,r0,band_HIGH_PAIRED__Z2
        - aux_cap_b9,r0,band_HIGH_PAIRED
        + 1.8253501688 aux_cap_helper__b9,r0,band_HIGH_PAIRED__condX3
        + 2.0281668543 aux_cap_helper__b9,r0,band_HIGH_PAIRED__condX1
        + 2.8394335959 aux_cap_helper__b9,r0,band_HIGH_PAIRED__condX2  = 0
 c3574: - aux_Omega_b7,r4 + 2844.94326768265 aux_sv_b7,r4  = 0
 c3575: aux_cap_helper__b9,r4,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b9,r4,band_HIGH_PAIRED__Z1 <= 0
 c3576: aux_cap_helper__b7,r7,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b7,r7,band_UNPAIRED__Z2 <= 0
 c3577: aux_cap_helper__b8,r0,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b8,r0,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b8,r0,band_HIGH_PAIRED__Z1  = 1
 c3578: - sv_function_helpervar_b9,r0__condX1 <= 0
 c3579: - 0.253841888 aux_cap_helper__b2,r10,band_UNPAIRED__Z2
        + 0.888446608 aux_cap_helper__b2,r10,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b2,r10,band_UNPAIRED__condX1
        - aux_cap_b2,r10,band_UNPAIRED  = 0
 c3580: aux_cap_helper__b0,r11,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b0,r11,band_HIGH_PAIRED__Z1 <= 0
 c3581: - X__b1,r5,band_UNPAIRED + aux_cap_helper__b1,r5,band_UNPAIRED__condX2
        + aux_cap_helper__b1,r5,band_UNPAIRED__condX1  = 0
 c3582: aux_cap_helper__b8,r1,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b8,r1,band_HIGH_PAIRED__Z2 <= 0
 c3583: - aux_cap_b9,r12,band_HIGH_PAIRED
        + 1.8253501688 aux_cap_helper__b9,r12,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b9,r12,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b9,r12,band_HIGH_PAIRED__condX1
        - 0.8112667416 aux_cap_helper__b9,r12,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b9,r12,band_HIGH_PAIRED__Z3  = 0
 c3584: aux_cap_helper__b5,r3,band_LOW_PAIRED__condX1
        - aux_cap_helper__b5,r3,band_LOW_PAIRED__Z1 <= 0
 c3585: 1.21690011250031 aux_cap_helper__b3,r9,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b3,r9,band_HIGH_PAIRED__Z2
        + 1.8253501688 aux_cap_helper__b3,r9,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b3,r9,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b3,r9,band_HIGH_PAIRED__condX1
        - aux_cap_b3,r9,band_HIGH_PAIRED  = 0
 c3586: aux_cap_helper__b9,r4,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b9,r4,band_HIGH_PAIRED__Z2 <= 0
 c3587: - 0.8112667416 aux_cap_helper__b9,r5,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b9,r5,band_HIGH_PAIRED__Z3
        - aux_cap_b9,r5,band_HIGH_PAIRED
        + 2.0281668543 aux_cap_helper__b9,r5,band_HIGH_PAIRED__condX1
        + 1.8253501688 aux_cap_helper__b9,r5,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b9,r5,band_HIGH_PAIRED__condX2  = 0
 c3588: - 0.8112667416 aux_cap_helper__b3,r4,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b3,r4,band_HIGH_PAIRED__Z3
        + 2.0281668543 aux_cap_helper__b3,r4,band_HIGH_PAIRED__condX1
        + 2.8394335959 aux_cap_helper__b3,r4,band_HIGH_PAIRED__condX2
        + 1.8253501688 aux_cap_helper__b3,r4,band_HIGH_PAIRED__condX3
        - aux_cap_b3,r4,band_HIGH_PAIRED  = 0
 c3589: aux_cap_helper__b9,r6,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b9,r6,band_UNPAIRED__Z2 <= 0
 c3590: X__b2,r2,band_LOW_PAIRED + X__b6,r2,band_LOW_PAIRED
        + X__b0,r2,band_LOW_PAIRED + X__b4,r2,band_LOW_PAIRED
        + X__b3,r2,band_LOW_PAIRED + X__b1,r2,band_LOW_PAIRED
        + X__b5,r2,band_LOW_PAIRED + X__b7,r2,band_LOW_PAIRED
        + X__b8,r2,band_LOW_PAIRED + X__b9,r2,band_LOW_PAIRED <= 2
 c3591: aux_cap_helper__b9,r11,band_UNPAIRED__condX1
        + aux_cap_helper__b9,r11,band_UNPAIRED__condX2
        - X__b9,r11,band_UNPAIRED  = 0
 c3592: 1824.44485328898 aux_sv_b0,r4 - aux_Omega_b0,r4  = 0
 c3593: aux_cap_helper__b0,r11,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b0,r11,band_HIGH_PAIRED__Z2 <= 0
 c3594: aux_cap_helper__b5,r3,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b5,r3,band_LOW_PAIRED__Z2 <= 0
 c3595: - aux_cap_helper__b7,r11,band_HIGH_PAIRED__condX1 <= 0
 c3596: aux_cap_helper__b0,r3,band_LOW_PAIRED__condX1
        - aux_cap_helper__b0,r3,band_LOW_PAIRED__Z1 <= 0
 c3597: aux_Omega_b9,r1 + aux_Omega_b9,r0 + aux_Omega_b9,r3 + aux_Omega_b9,r2
        + aux_Omega_b9,r5 + aux_Omega_b9,r4 + aux_Omega_b9,r7 + aux_Omega_b9,r6
        + aux_Omega_b9,r9 + aux_Omega_b9,r8 - GlobalBidder_PSI__i9,k1
        + aux_Omega_b9,r13 + aux_Omega_b9,r10 + aux_Omega_b9,r11
        + aux_Omega_b9,r12 - 20668026.8748422 GlobalBidder_hat{W}__i9,k1
        >= -20668026.8748422
 c3598: aux_cap_helper__b4,r13,band_UNPAIRED__Z2
        - aux_cap_helper__b4,r13,band_UNPAIRED__condX2 <= 0
 c3599: - aux_cap_helper__b8,r7,band_HIGH_PAIRED__condX1 <= 0
 c3600: sv_function_helpervar_b2,r3__condX1
        - 0.158453086903001 sv_function_helpervar_b2,r3__Z1 <= 0
 c3601: 315.435396069556 sv_function_helpervar_b6,r5__Z3
        + 142.295533015367 sv_function_helpervar_b6,r5__Z2
        + 942.4308567182 sv_function_helpervar_b6,r5__condX1
        + 14.9976675656 sv_function_helpervar_b6,r5__condX3
        + 44.401441158 sv_function_helpervar_b6,r5__condX2 - aux_sv_b6,r5  = 0
 c3602: - aux_cap_helper__b2,r2,band_HIGH_PAIRED__condX1 <= 0
 c3603: - 5.74218277061943 sv_function_helpervar_b4,r12__Z2
        + sv_function_helpervar_b4,r12__condX2 <= 0
 c3604: aux_cap_helper__b7,r9,band_UNPAIRED__condX2
        + aux_cap_helper__b7,r9,band_UNPAIRED__condX1 - X__b7,r9,band_UNPAIRED
         = 0
 c3605: aux_cap_helper__b1,r11,band_LOW_PAIRED__condX1
        - aux_cap_helper__b1,r11,band_LOW_PAIRED__Z1 <= 0
 c3606: - aux_Omega_b0,r12 + 1080.49075135384 aux_sv_b0,r12  = 0
 c3607: 1.21690011250031 aux_cap_helper__b4,r12,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b4,r12,band_HIGH_PAIRED__Z2
        - aux_cap_b4,r12,band_HIGH_PAIRED
        + 1.8253501688 aux_cap_helper__b4,r12,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b4,r12,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b4,r12,band_HIGH_PAIRED__condX1  = 0
 c3608: - 2 aux_cap_helper__b6,r2,band_LOW_PAIRED__Z2
        + aux_cap_helper__b6,r2,band_LOW_PAIRED__condX2 <= 0
 c3609: sv_function_helpervar_b6,r0__Z1 + sv_function_helpervar_b6,r0__Z2
        + sv_function_helpervar_b6,r0__Z3  = 1
 c3610: - 3 aux_cap_helper__b2,r11,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b2,r11,band_HIGH_PAIRED__condX3 <= 0
 c3611: aux_cap_helper__b7,r7,band_UNPAIRED__condX1
        - aux_cap_helper__b7,r7,band_UNPAIRED__Z1 <= 0
 c3612: sv_function_helpervar_b9,r3__condX3
        - 15.8453086887446 sv_function_helpervar_b9,r3__Z3 <= 0
 c3613: - 0.253841888 aux_cap_helper__b8,r10,band_UNPAIRED__Z2
        + 0.888446608 aux_cap_helper__b8,r10,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b8,r10,band_UNPAIRED__condX1
        - aux_cap_b8,r10,band_UNPAIRED  = 0
 c3614: aux_cap_helper__b1,r0,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b1,r0,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b1,r0,band_HIGH_PAIRED__Z1  = 1
 c3615: - 2 aux_cap_helper__b2,r8,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b2,r8,band_HIGH_PAIRED__condX2 <= 0
 c3616: 0.888446608 aux_cap_helper__b3,r10,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b3,r10,band_UNPAIRED__condX1
        - aux_cap_b3,r10,band_UNPAIRED
        - 0.253841888 aux_cap_helper__b3,r10,band_UNPAIRED__Z2  = 0
 c3617: - 2 aux_cap_helper__b3,r11,band_UNPAIRED__Z2
        + aux_cap_helper__b3,r11,band_UNPAIRED__condX2 <= 0
 c3618: - 1.2715511236 aux_cap_helper__b6,r2,band_LOW_PAIRED__Z2
        + 4.4504289328 aux_cap_helper__b6,r2,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b6,r2,band_LOW_PAIRED__condX1
        - aux_cap_b6,r2,band_LOW_PAIRED  = 0
 c3619: aux_cap_helper__b9,r6,band_UNPAIRED__condX1
        - aux_cap_helper__b9,r6,band_UNPAIRED__Z1 <= 0
 c3620: - aux_cap_b0,r1,band_LOW_PAIRED
        + 4.4504289328 aux_cap_helper__b0,r1,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b0,r1,band_LOW_PAIRED__condX1
        - 1.2715511236 aux_cap_helper__b0,r1,band_LOW_PAIRED__Z2  = 0
 c3621: aux_cap_helper__b6,r4,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b6,r4,band_HIGH_PAIRED__Z1 <= 0
 c3622: - 2 aux_cap_helper__b0,r9,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b0,r9,band_HIGH_PAIRED__condX2 <= 0
 c3623: - 2 aux_cap_helper__b2,r10,band_LOW_PAIRED__Z2
        + aux_cap_helper__b2,r10,band_LOW_PAIRED__condX2 <= 0
 c3624: aux_cap_helper__b7,r8,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b7,r8,band_HIGH_PAIRED__Z3 <= 0
 c3625: - sv_function_helpervar_b1,r4__condX3
        + 5.93643512983155 sv_function_helpervar_b1,r4__Z3 <= 0
 c3626: aux_cap_helper__b9,r3,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b9,r3,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b9,r3,band_HIGH_PAIRED__condX1
        - X__b9,r3,band_HIGH_PAIRED  = 0
 c3627: - X__b0,r1,band_HIGH_PAIRED
        + aux_cap_helper__b0,r1,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b0,r1,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b0,r1,band_HIGH_PAIRED__condX3  = 0
 c3628: aux_cap_helper__b2,r1,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b2,r1,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b2,r1,band_HIGH_PAIRED__Z3  = 1
 c3629: aux_cap_helper__b6,r4,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b6,r4,band_HIGH_PAIRED__Z2 <= 0
 c3630: - aux_cap_helper__b6,r2,band_LOW_PAIRED__Z1
        + aux_cap_helper__b6,r2,band_LOW_PAIRED__condX1 <= 0
 c3631: - 0.253841888 aux_cap_helper__b9,r7,band_UNPAIRED__Z2
        + 0.888446608 aux_cap_helper__b9,r7,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b9,r7,band_UNPAIRED__condX1
        - aux_cap_b9,r7,band_UNPAIRED  = 0
 c3632: aux_cap_helper__b3,r0,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b3,r0,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b3,r0,band_HIGH_PAIRED__Z3  = 1
 c3633: - 0.15845308694837 sv_function_helpervar_b0,r4__Z1
        + sv_function_helpervar_b0,r4__condX1 <= 0
 c3634: sv_function_helpervar_b5,r5__Z1 + sv_function_helpervar_b5,r5__Z2
        + sv_function_helpervar_b5,r5__Z3  = 1
 c3635: sv_function_helpervar_b9,r8__Z1 + sv_function_helpervar_b9,r8__Z2
        + sv_function_helpervar_b9,r8__Z3  = 1
 c3636: aux_cap_helper__b0,r3,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b0,r3,band_LOW_PAIRED__Z2 <= 0
 c3637: 0.158453086996438 sv_function_helpervar_b0,r6__Z2
        - sv_function_helpervar_b0,r6__condX2 <= 0
 c3638: - aux_cap_helper__b2,r8,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b2,r8,band_HIGH_PAIRED__condX1 <= 0
 c3639: - aux_cap_helper__b3,r6,band_UNPAIRED__condX2
        + aux_cap_helper__b3,r6,band_UNPAIRED__Z2 <= 0
 c3640: aux_cap_b2,r12,band_LOW_PAIRED + aux_cap_b2,r12,band_UNPAIRED
        + aux_cap_b2,r12,band_HIGH_PAIRED - aux_c_b2,r12  = 0
 c3641: - aux_cap_helper__b3,r4,band_LOW_PAIRED__condX2
        + aux_cap_helper__b3,r4,band_LOW_PAIRED__Z2 <= 0
 c3642: sv_function_helpervar_b8,r0__condX2
        - 8.01571555507819 sv_function_helpervar_b8,r0__Z2 <= 0
 c3643: aux_cap_helper__b4,r7,band_LOW_PAIRED__Z1
        + aux_cap_helper__b4,r7,band_LOW_PAIRED__Z2  = 1
 c3644: - sv_function_helpervar_b4,r0__condX1 <= 0
 c3645: - X__b1,r1,band_LOW_PAIRED
        + aux_cap_helper__b1,r1,band_LOW_PAIRED__condX2
        + aux_cap_helper__b1,r1,band_LOW_PAIRED__condX1  = 0
 c3646: aux_cap_helper__b1,r10,band_UNPAIRED__Z1
        + aux_cap_helper__b1,r10,band_UNPAIRED__Z2  = 1
 c3647: - 0.158453086941154 sv_function_helpervar_b6,r8__Z1
        + sv_function_helpervar_b6,r8__condX1 <= 0
 c3648: sv_function_helpervar_b2,r1__Z2 + sv_function_helpervar_b2,r1__Z3
        + sv_function_helpervar_b2,r1__Z1  = 1
 c3649: - X__b0,r2,band_LOW_PAIRED
        + aux_cap_helper__b0,r2,band_LOW_PAIRED__condX2
        + aux_cap_helper__b0,r2,band_LOW_PAIRED__condX1  = 0
 c3650: - aux_cap_helper__b5,r12,band_UNPAIRED__condX1 <= 0
 c3651: X__b8,r12,band_UNPAIRED + X__b8,r12,band_HIGH_PAIRED
        - GlobalBidder_W_ir__b8,r12 + X__b8,r12,band_LOW_PAIRED >= 0
 c3652: aux_cap_helper__b4,r2,band_UNPAIRED__Z2
        + aux_cap_helper__b4,r2,band_UNPAIRED__Z1  = 1
 c3653: - aux_cap_helper__b1,r6,band_HIGH_PAIRED__condX1 <= 0
 c3654: - aux_cap_helper__b5,r5,band_UNPAIRED__condX2
        + aux_cap_helper__b5,r5,band_UNPAIRED__Z2 <= 0
 c3655: - v__7 + 0.80684 GlobalBidder_PSI__i7,k2
        + 0.95171 GlobalBidder_PSI__i7,k1 + GlobalBidder_PSI__i7,k0  = 0
 c3656: - aux_cap_helper__b5,r8,band_LOW_PAIRED__condX1 <= 0
 c3657: - 1.2715511236 aux_cap_helper__b8,r10,band_LOW_PAIRED__Z2
        + 3.1788778092 aux_cap_helper__b8,r10,band_LOW_PAIRED__condX1
        + 4.4504289328 aux_cap_helper__b8,r10,band_LOW_PAIRED__condX2
        - aux_cap_b8,r10,band_LOW_PAIRED  = 0
 c3658: 0.888446608 aux_cap_helper__b7,r10,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b7,r10,band_UNPAIRED__condX1
        - aux_cap_b7,r10,band_UNPAIRED
        - 0.253841888 aux_cap_helper__b7,r10,band_UNPAIRED__Z2  = 0
 c3659: aux_cap_helper__b8,r12,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b8,r12,band_HIGH_PAIRED__condX2 <= 0
 c3660: - aux_cap_helper__b8,r1,band_LOW_PAIRED__condX1 <= 0
 c3661: - aux_cap_b1,r7,band_LOW_PAIRED
        + 3.1788778092 aux_cap_helper__b1,r7,band_LOW_PAIRED__condX1
        + 4.4504289328 aux_cap_helper__b1,r7,band_LOW_PAIRED__condX2
        - 1.2715511236 aux_cap_helper__b1,r7,band_LOW_PAIRED__Z2  = 0
 c3662: - aux_cap_helper__b5,r2,band_HIGH_PAIRED__condX1 <= 0
 c3663: 2 aux_cap_helper__b8,r12,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b8,r12,band_HIGH_PAIRED__condX3 <= 0
 c3664: - aux_cap_helper__b3,r4,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b3,r4,band_HIGH_PAIRED__condX1 <= 0
 c3665: sv_function_helpervar_b6,r12__condX3
        - 15.8453086887446 sv_function_helpervar_b6,r12__Z3 <= 0
 c3666: GlobalBidder_W_ir__b7,r3 + GlobalBidder_W_ir__b7,r2
        + GlobalBidder_W_ir__b7,r1 + GlobalBidder_W_ir__b7,r0
        + GlobalBidder_W_ir__b7,r7 + GlobalBidder_W_ir__b7,r6
        + GlobalBidder_W_ir__b7,r5 + GlobalBidder_W_ir__b7,r4
        + GlobalBidder_W_ir__b7,r9 + GlobalBidder_W_ir__b7,r8
        - GlobalBidder_W_i_7 + GlobalBidder_W_ir__b7,r13
        + GlobalBidder_W_ir__b7,r11 + GlobalBidder_W_ir__b7,r12
        + GlobalBidder_W_ir__b7,r10  = 0
 c3667: - aux_cap_helper__b4,r9,band_UNPAIRED__condX1 <= 0
 c3668: - aux_cap_b5,r6,band_HIGH_PAIRED
        - 0.8112667416 aux_cap_helper__b5,r6,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b5,r6,band_HIGH_PAIRED__Z3
        + 1.8253501688 aux_cap_helper__b5,r6,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b5,r6,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b5,r6,band_HIGH_PAIRED__condX1  = 0
 c3669: - aux_cap_helper__b6,r8,band_UNPAIRED__condX2
        + aux_cap_helper__b6,r8,band_UNPAIRED__Z2 <= 0
 c3670: sv_function_helpervar_b9,r4__Z1 + sv_function_helpervar_b9,r4__Z2
        + sv_function_helpervar_b9,r4__Z3  = 1
 c3671: sv_function_helpervar_b7,r6__condX3
        - 15.8453086887446 sv_function_helpervar_b7,r6__Z3 <= 0
 c3672: 0.0102040816326531 X__b7,r3,band_LOW_PAIRED - GlobalBidder_W_ir__b7,r3
        + 0.0102040816326531 X__b7,r3,band_UNPAIRED
        + 0.0102040816326531 X__b7,r3,band_HIGH_PAIRED <= 0
 c3673: 6.52004106046921 sv_function_helpervar_b0,r13__Z3
        - sv_function_helpervar_b0,r13__condX3 <= 0
 c3674: aux_cap_helper__b5,r12,band_LOW_PAIRED__Z1
        + aux_cap_helper__b5,r12,band_LOW_PAIRED__Z2  = 1
 c3675: - 1.2715511236 aux_cap_helper__b2,r9,band_LOW_PAIRED__Z2
        - aux_cap_b2,r9,band_LOW_PAIRED
        + 3.1788778092 aux_cap_helper__b2,r9,band_LOW_PAIRED__condX1
        + 4.4504289328 aux_cap_helper__b2,r9,band_LOW_PAIRED__condX2  = 0
 c3676: - 0.8112667416 aux_cap_helper__b7,r3,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b7,r3,band_HIGH_PAIRED__Z3
        + 1.8253501688 aux_cap_helper__b7,r3,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b7,r3,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b7,r3,band_HIGH_PAIRED__condX1
        - aux_cap_b7,r3,band_HIGH_PAIRED  = 0
 c3677: - aux_cap_b7,r10,band_HIGH_PAIRED
        + 2.0281668543 aux_cap_helper__b7,r10,band_HIGH_PAIRED__condX1
        + 1.8253501688 aux_cap_helper__b7,r10,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b7,r10,band_HIGH_PAIRED__condX2
        - 0.8112667416 aux_cap_helper__b7,r10,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b7,r10,band_HIGH_PAIRED__Z3  = 0
 c3678: aux_cap_helper__b4,r6,band_LOW_PAIRED__Z2
        - aux_cap_helper__b4,r6,band_LOW_PAIRED__condX2 <= 0
 c3679: - aux_cap_b0,r3,band_LOW_PAIRED
        + 3.1788778092 aux_cap_helper__b0,r3,band_LOW_PAIRED__condX1
        + 4.4504289328 aux_cap_helper__b0,r3,band_LOW_PAIRED__condX2
        - 1.2715511236 aux_cap_helper__b0,r3,band_LOW_PAIRED__Z2  = 0
 c3680: - aux_cap_helper__b3,r11,band_UNPAIRED__Z1
        + aux_cap_helper__b3,r11,band_UNPAIRED__condX1 <= 0
 c3681: - sv_function_helpervar_b9,r9__condX1 <= 0
 c3682: - sv_function_helpervar_b9,r4__condX3
        + 7.52273391768105 sv_function_helpervar_b9,r4__Z3 <= 0
 c3683: - aux_Omega_b7,r13 + 51.5088946637852 aux_sv_b7,r13  = 0
 c3684: - sv_function_helpervar_b5,r13__condX1 <= 0
 c3685: sv_function_helpervar_b2,r2__condX2
        - 5.80744827810344 sv_function_helpervar_b2,r2__Z2 <= 0
 c3686: sv_function_helpervar_b5,r9__Z1 + sv_function_helpervar_b5,r9__Z2
        + sv_function_helpervar_b5,r9__Z3  = 1
 c3687: aux_cap_helper__b1,r11,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b1,r11,band_LOW_PAIRED__Z2 <= 0
 c3688: - aux_cap_helper__b0,r11,band_UNPAIRED__condX1 <= 0
 c3689: aux_cap_helper__b0,r7,band_UNPAIRED__Z1
        + aux_cap_helper__b0,r7,band_UNPAIRED__Z2  = 1
 c3690: - sv_function_helpervar_b8,r12__condX3
        + 6.2737382343784 sv_function_helpervar_b8,r12__Z3 <= 0
 c3691: - aux_cap_helper__b2,r10,band_LOW_PAIRED__Z1
        + aux_cap_helper__b2,r10,band_LOW_PAIRED__condX1 <= 0
 c3692: - aux_cap_helper__b0,r8,band_LOW_PAIRED__condX2
        + aux_cap_helper__b0,r8,band_LOW_PAIRED__Z2 <= 0
 c3693: - 20668026.8748422 GlobalBidder_hat{W}__i8,k1 - GlobalBidder_PSI__i8,k1
        + aux_Omega_b8,r12 + aux_Omega_b8,r13 + aux_Omega_b8,r10
        + aux_Omega_b8,r11 + aux_Omega_b8,r5 + aux_Omega_b8,r6
        + aux_Omega_b8,r7 + aux_Omega_b8,r8 + aux_Omega_b8,r9 + aux_Omega_b8,r0
        + aux_Omega_b8,r1 + aux_Omega_b8,r2 + aux_Omega_b8,r3 + aux_Omega_b8,r4
        >= -20668026.8748422
 c3694: aux_cap_helper__b2,r0,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b2,r0,band_UNPAIRED__Z2 <= 0
 c3695: - aux_cap_helper__b1,r13,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b1,r13,band_HIGH_PAIRED__Z2 <= 0
 c3696: aux_cap_helper__b7,r10,band_UNPAIRED__condX2
        + aux_cap_helper__b7,r10,band_UNPAIRED__condX1
        - X__b7,r10,band_UNPAIRED  = 0
 c3697: - aux_cap_helper__b0,r8,band_UNPAIRED__Z1
        + aux_cap_helper__b0,r8,band_UNPAIRED__condX1 <= 0
 c3698: - X__b2,r8,band_HIGH_PAIRED
        + aux_cap_helper__b2,r8,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b2,r8,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b2,r8,band_HIGH_PAIRED__condX3  = 0
 c3699: - aux_cap_b8,r8,band_LOW_PAIRED
        - 1.2715511236 aux_cap_helper__b8,r8,band_LOW_PAIRED__Z2
        + 3.1788778092 aux_cap_helper__b8,r8,band_LOW_PAIRED__condX1
        + 4.4504289328 aux_cap_helper__b8,r8,band_LOW_PAIRED__condX2  = 0
 c3700: aux_cap_b5,r10,band_HIGH_PAIRED + aux_cap_b5,r10,band_UNPAIRED
        - aux_c_b5,r10 + aux_cap_b5,r10,band_LOW_PAIRED  = 0
 c3701: - 6.5034000540409 sv_function_helpervar_b1,r8__Z2
        + sv_function_helpervar_b1,r8__condX2 <= 0
 c3702: - sv_function_helpervar_b1,r2__condX1 <= 0
 c3703: aux_cap_helper__b6,r12,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b6,r12,band_HIGH_PAIRED__Z3 <= 0
 c3704: sv_function_helpervar_b8,r7__condX1
        + sv_function_helpervar_b8,r7__condX3
        + sv_function_helpervar_b8,r7__condX2 - aux_c_b8,r7  = 0
 c3705: - aux_c_b5,r11 + sv_function_helpervar_b5,r11__condX1
        + sv_function_helpervar_b5,r11__condX3
        + sv_function_helpervar_b5,r11__condX2  = 0
 c3706: - aux_cap_helper__b1,r13,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b1,r13,band_HIGH_PAIRED__Z3 <= 0
 c3707: - 2 aux_cap_helper__b0,r8,band_UNPAIRED__Z2
        + aux_cap_helper__b0,r8,band_UNPAIRED__condX2 <= 0
 c3708: - 0.253841888 aux_cap_helper__b6,r9,band_UNPAIRED__Z2
        + 0.63460472 aux_cap_helper__b6,r9,band_UNPAIRED__condX1
        + 0.888446608 aux_cap_helper__b6,r9,band_UNPAIRED__condX2
        - aux_cap_b6,r9,band_UNPAIRED  = 0
 c3709: - 0.15845308689803 sv_function_helpervar_b1,r0__Z1
        + sv_function_helpervar_b1,r0__condX1 <= 0
 c3710: - aux_cap_helper__b3,r9,band_LOW_PAIRED__Z1
        + aux_cap_helper__b3,r9,band_LOW_PAIRED__condX1 <= 0
 c3711: aux_cap_helper__b4,r12,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b4,r12,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b4,r12,band_HIGH_PAIRED__Z1  = 1
 c3712: - sv_function_helpervar_b4,r9__condX1 <= 0
 c3713: sv_function_helpervar_b3,r2__condX3
        + sv_function_helpervar_b3,r2__condX2
        + sv_function_helpervar_b3,r2__condX1 - aux_c_b3,r2  = 0
 c3714: 96.5342721713339 aux_sv_b0,r0 - aux_Omega_b0,r0  = 0
 c3715: 5.91034954743343 sv_function_helpervar_b4,r0__Z3
        - sv_function_helpervar_b4,r0__condX3 <= 0
 c3716: - aux_cap_helper__b4,r12,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b4,r12,band_HIGH_PAIRED__condX1 <= 0
 c3717: - aux_c_b8,r9 + aux_cap_b8,r9,band_UNPAIRED
        + aux_cap_b8,r9,band_HIGH_PAIRED + aux_cap_b8,r9,band_LOW_PAIRED  = 0
 c3718: aux_cap_helper__b7,r5,band_LOW_PAIRED__Z2
        + aux_cap_helper__b7,r5,band_LOW_PAIRED__Z1  = 1
 c3719: - X__b3,r12,band_HIGH_PAIRED
        + aux_cap_helper__b3,r12,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b3,r12,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b3,r12,band_HIGH_PAIRED__condX1  = 0
 c3720: aux_cap_helper__b2,r0,band_UNPAIRED__condX1
        - aux_cap_helper__b2,r0,band_UNPAIRED__Z1 <= 0
 c3721: - aux_cap_b2,r3,band_UNPAIRED
        - 0.253841888 aux_cap_helper__b2,r3,band_UNPAIRED__Z2
        + 0.888446608 aux_cap_helper__b2,r3,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b2,r3,band_UNPAIRED__condX1  = 0
 c3722: aux_cap_helper__b0,r9,band_LOW_PAIRED__Z1
        + aux_cap_helper__b0,r9,band_LOW_PAIRED__Z2  = 1
 c3723: 5.38866541022754 sv_function_helpervar_b3,r11__Z3
        - sv_function_helpervar_b3,r11__condX3 <= 0
 c3724: - 0.158453086901135 sv_function_helpervar_b2,r6__Z1
        + sv_function_helpervar_b2,r6__condX1 <= 0
 c3725: - aux_cap_helper__b3,r2,band_LOW_PAIRED__Z1
        + aux_cap_helper__b3,r2,band_LOW_PAIRED__condX1 <= 0
 c3726: - sv_function_helpervar_b2,r5__condX1 <= 0
 c3727: aux_cap_helper__b7,r11,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b7,r11,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b7,r11,band_HIGH_PAIRED__condX2
        - X__b7,r11,band_HIGH_PAIRED  = 0
 c3728: - aux_cap_helper__b8,r11,band_HIGH_PAIRED__condX1 <= 0
 c3729: - 2 aux_cap_helper__b8,r0,band_UNPAIRED__Z2
        + aux_cap_helper__b8,r0,band_UNPAIRED__condX2 <= 0
 c3730: sv_function_helpervar_b3,r3__Z2 + sv_function_helpervar_b3,r3__Z1
        + sv_function_helpervar_b3,r3__Z3  = 1
 c3731: 14.5239218122 sv_function_helpervar_b6,r1__condX3
        + 47.0694015669 sv_function_helpervar_b6,r1__condX2
        + 942.4308565814 sv_function_helpervar_b6,r1__condX1
        + 141.872786451787 sv_function_helpervar_b6,r1__Z2
        + 322.94204377262 sv_function_helpervar_b6,r1__Z3 - aux_sv_b6,r1  = 0
 c3732: - aux_c_b7,r0 + sv_function_helpervar_b7,r0__condX1
        + sv_function_helpervar_b7,r0__condX2
        + sv_function_helpervar_b7,r0__condX3  = 0
 c3733: - aux_cap_helper__b4,r1,band_LOW_PAIRED__condX1 <= 0
 c3734: aux_cap_helper__b2,r3,band_LOW_PAIRED__Z2
        - aux_cap_helper__b2,r3,band_LOW_PAIRED__condX2 <= 0
 c3735: - X__b6,r10,band_HIGH_PAIRED
        + aux_cap_helper__b6,r10,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b6,r10,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b6,r10,band_HIGH_PAIRED__condX2  = 0
 c3736: aux_cap_helper__b1,r7,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b1,r7,band_HIGH_PAIRED__condX2 <= 0
 c3737: - 2 aux_cap_helper__b3,r1,band_UNPAIRED__Z2
        + aux_cap_helper__b3,r1,band_UNPAIRED__condX2 <= 0
 c3738: - aux_cap_helper__b5,r10,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b5,r10,band_HIGH_PAIRED__condX1 <= 0
 c3739: aux_cap_helper__b6,r8,band_LOW_PAIRED__Z1
        + aux_cap_helper__b6,r8,band_LOW_PAIRED__Z2  = 1
 c3740: 4.4504289328 aux_cap_helper__b6,r5,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b6,r5,band_LOW_PAIRED__condX1
        - 1.2715511236 aux_cap_helper__b6,r5,band_LOW_PAIRED__Z2
        - aux_cap_b6,r5,band_LOW_PAIRED  = 0
 c3741: - aux_cap_helper__b4,r3,band_HIGH_PAIRED__condX1 <= 0
 c3742: aux_Omega_b1,r6 + aux_Omega_b1,r1 + aux_Omega_b1,r0 - v__1  = 0
 c3743: aux_cap_helper__b7,r13,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b7,r13,band_HIGH_PAIRED__Z3 <= 0
 c3744: - sv_function_helpervar_b1,r7__condX1 <= 0
 c3745: sv_function_helpervar_b9,r0__Z1 + sv_function_helpervar_b9,r0__Z2
        + sv_function_helpervar_b9,r0__Z3  = 1
 c3746: aux_cap_helper__b1,r9,band_LOW_PAIRED__condX1
        + aux_cap_helper__b1,r9,band_LOW_PAIRED__condX2
        - X__b1,r9,band_LOW_PAIRED  = 0
 c3747: - aux_cap_helper__b6,r9,band_HIGH_PAIRED__condX1 <= 0
 c3748: - 3 aux_cap_helper__b5,r0,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b5,r0,band_HIGH_PAIRED__condX3 <= 0
 c3749: - aux_cap_helper__b1,r7,band_HIGH_PAIRED__condX1 <= 0
 c3750: 447.447128613989 aux_sv_b5,r10 - aux_Omega_b5,r10  = 0
 c3751: - 15.8453086887446 sv_function_helpervar_b9,r9__Z3
        + sv_function_helpervar_b9,r9__condX3 <= 0
 c3752: aux_cap_helper__b7,r6,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b7,r6,band_HIGH_PAIRED__Z3 <= 0
 c3753: aux_cap_helper__b7,r0,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b7,r0,band_HIGH_PAIRED__condX2 <= 0
 c3754: aux_cap_helper__b5,r6,band_LOW_PAIRED__condX1
        + aux_cap_helper__b5,r6,band_LOW_PAIRED__condX2
        - X__b5,r6,band_LOW_PAIRED  = 0
 c3755: aux_cap_helper__b0,r5,band_LOW_PAIRED__Z2
        - aux_cap_helper__b0,r5,band_LOW_PAIRED__condX2 <= 0
 c3756: aux_cap_helper__b2,r4,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b2,r4,band_HIGH_PAIRED__Z3 <= 0
 c3757: - sv_function_helpervar_b9,r8__condX1 <= 0
 c3758: - aux_cap_helper__b9,r6,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b9,r6,band_HIGH_PAIRED__Z2 <= 0
 c3759: - sv_function_helpervar_b6,r7__condX2
        + 0.158453086904266 sv_function_helpervar_b6,r7__Z2 <= 0
 c3760: aux_cap_helper__b3,r5,band_LOW_PAIRED__Z1
        + aux_cap_helper__b3,r5,band_LOW_PAIRED__Z2  = 1
 c3761: sv_function_helpervar_b4,r2__condX3
        - 15.8453086887446 sv_function_helpervar_b4,r2__Z3 <= 0
 c3762: aux_cap_helper__b0,r4,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b0,r4,band_HIGH_PAIRED__Z1 <= 0
 c3763: aux_cap_helper__b4,r6,band_UNPAIRED__condX1
        - aux_cap_helper__b4,r6,band_UNPAIRED__Z1 <= 0
 c3764: - 0.158453086947678 sv_function_helpervar_b4,r7__Z1
        + sv_function_helpervar_b4,r7__condX1 <= 0
 c3765: aux_cap_helper__b8,r11,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b8,r11,band_HIGH_PAIRED__condX2 <= 0
 c3766: 7.50122203625723 sv_function_helpervar_b8,r13__Z3
        - sv_function_helpervar_b8,r13__condX3 <= 0
 c3767: 0.15845308689553 sv_function_helpervar_b6,r4__Z2
        - sv_function_helpervar_b6,r4__condX2 <= 0
 c3768: aux_cap_helper__b8,r3,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b8,r3,band_HIGH_PAIRED__Z1 <= 0
 c3769: - X__b5,r4,band_UNPAIRED + aux_cap_helper__b5,r4,band_UNPAIRED__condX2
        + aux_cap_helper__b5,r4,band_UNPAIRED__condX1  = 0
 c3770: - aux_cap_helper__b4,r3,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b4,r3,band_HIGH_PAIRED__Z2 <= 0
 c3771: aux_Omega_b7,r10 + aux_Omega_b7,r11 + aux_Omega_b7,r12
        + aux_Omega_b7,r13 + aux_Omega_b7,r0 + aux_Omega_b7,r1
        + aux_Omega_b7,r2 + aux_Omega_b7,r3 + aux_Omega_b7,r4 + aux_Omega_b7,r5
        + aux_Omega_b7,r6 + aux_Omega_b7,r7 + aux_Omega_b7,r8 + aux_Omega_b7,r9
        - GlobalBidder_PSI__i7,k1 + 20668026.8748422 GlobalBidder_hat{W}__i7,k1
        <= 20668026.8748422
 c3772: aux_cap_helper__b3,r11,band_LOW_PAIRED__Z2
        + aux_cap_helper__b3,r11,band_LOW_PAIRED__Z1  = 1
 c3773: - sv_function_helpervar_b4,r8__condX3
        + 5.84349771088166 sv_function_helpervar_b4,r8__Z3 <= 0
 c3774: aux_cap_helper__b6,r9,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b6,r9,band_HIGH_PAIRED__condX2 <= 0
 c3775: 0.63460472 aux_cap_helper__b8,r8,band_UNPAIRED__condX1
        + 0.888446608 aux_cap_helper__b8,r8,band_UNPAIRED__condX2
        - 0.253841888 aux_cap_helper__b8,r8,band_UNPAIRED__Z2
        - aux_cap_b8,r8,band_UNPAIRED  = 0
 c3776: 0.63460472 aux_cap_helper__b5,r6,band_UNPAIRED__condX1
        + 0.888446608 aux_cap_helper__b5,r6,band_UNPAIRED__condX2
        - aux_cap_b5,r6,band_UNPAIRED
        - 0.253841888 aux_cap_helper__b5,r6,band_UNPAIRED__Z2  = 0
 c3777: 0.888446608 aux_cap_helper__b2,r4,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b2,r4,band_UNPAIRED__condX1
        - 0.253841888 aux_cap_helper__b2,r4,band_UNPAIRED__Z2
        - aux_cap_b2,r4,band_UNPAIRED  = 0
 c3778: aux_cap_helper__b2,r11,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b2,r11,band_UNPAIRED__Z2 <= 0
 c3779: - 15.8453086887446 sv_function_helpervar_b3,r4__Z3
        + sv_function_helpervar_b3,r4__condX3 <= 0
 c3780: aux_cap_helper__b1,r4,band_LOW_PAIRED__condX1
        - aux_cap_helper__b1,r4,band_LOW_PAIRED__Z1 <= 0
 c3781: - X__b0,r12,band_UNPAIRED
        + aux_cap_helper__b0,r12,band_UNPAIRED__condX1
        + aux_cap_helper__b0,r12,band_UNPAIRED__condX2  = 0
 c3782: - aux_cap_b0,r9,band_UNPAIRED
        - 0.253841888 aux_cap_helper__b0,r9,band_UNPAIRED__Z2
        + 0.63460472 aux_cap_helper__b0,r9,band_UNPAIRED__condX1
        + 0.888446608 aux_cap_helper__b0,r9,band_UNPAIRED__condX2  = 0
 c3783: - aux_cap_helper__b3,r13,band_HIGH_PAIRED__condX1 <= 0
 c3784: aux_cap_helper__b5,r13,band_LOW_PAIRED__Z1
        + aux_cap_helper__b5,r13,band_LOW_PAIRED__Z2  = 1
 c3785: - aux_sv_b5,r12 + 147.791389968633 sv_function_helpervar_b5,r12__Z2
        + 327.85222856637 sv_function_helpervar_b5,r12__Z3
        + 15.5648176391 sv_function_helpervar_b5,r12__condX3
        + 46.1880152369 sv_function_helpervar_b5,r12__condX2
        + 978.9018731467 sv_function_helpervar_b5,r12__condX1  = 0
 c3786: - X__b8,r6,band_HIGH_PAIRED
        + aux_cap_helper__b8,r6,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b8,r6,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b8,r6,band_HIGH_PAIRED__condX3  = 0
 c3787: aux_cap_b3,r5,band_UNPAIRED + aux_cap_b3,r5,band_LOW_PAIRED
        - aux_c_b3,r5 + aux_cap_b3,r5,band_HIGH_PAIRED  = 0
 c3788: sv_function_helpervar_b9,r10__condX1
        - 0.158453086914684 sv_function_helpervar_b9,r10__Z1 <= 0
 c3789: - sv_function_helpervar_b3,r3__condX1 <= 0
 c3790: - aux_cap_b4,r10,band_HIGH_PAIRED
        + 1.21690011250031 aux_cap_helper__b4,r10,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b4,r10,band_HIGH_PAIRED__Z2
        + 1.8253501688 aux_cap_helper__b4,r10,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b4,r10,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b4,r10,band_HIGH_PAIRED__condX1  = 0
 c3791: aux_cap_helper__b4,r8,band_LOW_PAIRED__condX1
        - aux_cap_helper__b4,r8,band_LOW_PAIRED__Z1 <= 0
 c3792: aux_cap_helper__b9,r5,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b9,r5,band_UNPAIRED__Z2 <= 0
 c3793: - 2 aux_cap_helper__b8,r2,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b8,r2,band_HIGH_PAIRED__condX2 <= 0
 c3794: - aux_cap_helper__b0,r12,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b0,r12,band_HIGH_PAIRED__condX1 <= 0
 c3795: aux_cap_helper__b2,r1,band_LOW_PAIRED__Z1
        + aux_cap_helper__b2,r1,band_LOW_PAIRED__Z2  = 1
 c3796: - aux_c_b8,r3 + aux_cap_b8,r3,band_UNPAIRED
        + aux_cap_b8,r3,band_HIGH_PAIRED + aux_cap_b8,r3,band_LOW_PAIRED  = 0
 c3797: aux_cap_helper__b5,r0,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b5,r0,band_LOW_PAIRED__Z2 <= 0
 c3798: aux_cap_helper__b7,r7,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b7,r7,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b7,r7,band_HIGH_PAIRED__Z1  = 1
 c3799: - 6.24134726058085 sv_function_helpervar_b5,r6__Z2
        + sv_function_helpervar_b5,r6__condX2 <= 0
 c3800: sv_function_helpervar_b6,r6__condX2
        + sv_function_helpervar_b6,r6__condX1
        + sv_function_helpervar_b6,r6__condX3 - aux_c_b6,r6  = 0
 c3801: 2 aux_cap_helper__b1,r8,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b1,r8,band_HIGH_PAIRED__condX3 <= 0
 c3802: - aux_cap_helper__b5,r3,band_UNPAIRED__condX1 <= 0
 c3803: - aux_cap_helper__b2,r4,band_UNPAIRED__condX2
        + aux_cap_helper__b2,r4,band_UNPAIRED__Z2 <= 0
 c3804: sv_function_helpervar_b9,r7__condX2
        - 7.41502217739354 sv_function_helpervar_b9,r7__Z2 <= 0
 c3805: aux_cap_helper__b4,r9,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b4,r9,band_HIGH_PAIRED__Z3 <= 0
 c3806: - aux_cap_helper__b8,r4,band_LOW_PAIRED__Z1
        + aux_cap_helper__b8,r4,band_LOW_PAIRED__condX1 <= 0
 c3807: X__b7,r6,band_LOW_PAIRED + X__b3,r6,band_LOW_PAIRED
        + X__b5,r6,band_LOW_PAIRED + X__b0,r6,band_LOW_PAIRED
        + X__b1,r6,band_LOW_PAIRED + X__b2,r6,band_LOW_PAIRED
        + X__b6,r6,band_LOW_PAIRED + X__b4,r6,band_LOW_PAIRED
        + X__b9,r6,band_LOW_PAIRED + X__b8,r6,band_LOW_PAIRED <= 2
 c3808: aux_cap_helper__b0,r2,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b0,r2,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b0,r2,band_HIGH_PAIRED__Z1  = 1
 c3809: - aux_cap_helper__b9,r3,band_LOW_PAIRED__condX1 <= 0
 c3810: - aux_cap_helper__b5,r7,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b5,r7,band_HIGH_PAIRED__condX1 <= 0
 c3811: - aux_cap_helper__b7,r0,band_UNPAIRED__Z1
        + aux_cap_helper__b7,r0,band_UNPAIRED__condX1 <= 0
 c3812: aux_cap_helper__b9,r3,band_LOW_PAIRED__Z1
        + aux_cap_helper__b9,r3,band_LOW_PAIRED__Z2  = 1
 c3813: - sv_function_helpervar_b9,r11__condX3
        + 7.82458034098202 sv_function_helpervar_b9,r11__Z3 <= 0
 c3814: - aux_cap_helper__b0,r4,band_UNPAIRED__condX1 <= 0
 c3815: 0.158453086901643 sv_function_helpervar_b3,r0__Z2
        - sv_function_helpervar_b3,r0__condX2 <= 0
 c3816: aux_cap_helper__b5,r1,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b5,r1,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b5,r1,band_HIGH_PAIRED__Z3  = 1
 c3817: 0.158453087015465 sv_function_helpervar_b9,r2__Z2
        - sv_function_helpervar_b9,r2__condX2 <= 0
 c3818: 259.156999087578 aux_sv_b4,r10 - aux_Omega_b4,r10  = 0
 c3819: - aux_c_b7,r3 + sv_function_helpervar_b7,r3__condX3
        + sv_function_helpervar_b7,r3__condX1
        + sv_function_helpervar_b7,r3__condX2  = 0
 c3820: 0.0102040816326531 X__b7,r5,band_UNPAIRED
        + 0.0102040816326531 X__b7,r5,band_HIGH_PAIRED
        - GlobalBidder_W_ir__b7,r5
        + 0.0102040816326531 X__b7,r5,band_LOW_PAIRED <= 0
 c3821: 151.45546655869 aux_sv_b0,r2 - aux_Omega_b0,r2  = 0
 c3822: - X__b0,r0,band_LOW_PAIRED
        + aux_cap_helper__b0,r0,band_LOW_PAIRED__condX2
        + aux_cap_helper__b0,r0,band_LOW_PAIRED__condX1  = 0
 c3823: - aux_cap_helper__b9,r6,band_HIGH_PAIRED__condX1 <= 0
 c3824: - aux_cap_helper__b5,r3,band_UNPAIRED__condX2
        + aux_cap_helper__b5,r3,band_UNPAIRED__Z2 <= 0
 c3825: aux_cap_helper__b9,r11,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b9,r11,band_LOW_PAIRED__Z2 <= 0
 c3826: - aux_cap_helper__b0,r5,band_LOW_PAIRED__condX1 <= 0
 c3827: - 3 aux_cap_helper__b1,r12,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b1,r12,band_HIGH_PAIRED__condX3 <= 0
 c3828: aux_cap_b0,r2,band_HIGH_PAIRED + aux_cap_b0,r2,band_UNPAIRED
        - aux_c_b0,r2 + aux_cap_b0,r2,band_LOW_PAIRED  = 0
 c3829: - aux_cap_helper__b9,r3,band_LOW_PAIRED__condX2
        + aux_cap_helper__b9,r3,band_LOW_PAIRED__Z2 <= 0
 c3830: sv_function_helpervar_b4,r9__condX1
        + sv_function_helpervar_b4,r9__condX3
        + sv_function_helpervar_b4,r9__condX2 - aux_c_b4,r9  = 0
 c3831: 132.938588037677 sv_function_helpervar_b4,r3__Z2
        + 303.488089732003 sv_function_helpervar_b4,r3__Z3
        + 883.4028076605 sv_function_helpervar_b4,r3__condX1
        + 13.5654399702 sv_function_helpervar_b4,r3__condX3
        + 44.4252236421 sv_function_helpervar_b4,r3__condX2 - aux_sv_b4,r3  = 0
 c3832: 398.227156668903 sv_function_helpervar_b3,r5__Z3
        + 173.23167727075 sv_function_helpervar_b3,r5__Z2 - aux_sv_b3,r5
        + 1152.1877363801 sv_function_helpervar_b3,r5__condX1
        + 58.9198132239 sv_function_helpervar_b3,r5__condX2
        + 17.5414394655 sv_function_helpervar_b3,r5__condX3  = 0
 c3833: - 6.00310871588683 sv_function_helpervar_b6,r10__Z2
        + sv_function_helpervar_b6,r10__condX2 <= 0
 c3834: aux_Omega_b7,r10 + aux_Omega_b7,r11 + aux_Omega_b7,r12
        + aux_Omega_b7,r13 + aux_Omega_b7,r0 + aux_Omega_b7,r1
        + aux_Omega_b7,r2 + aux_Omega_b7,r3 + aux_Omega_b7,r4 + aux_Omega_b7,r5
        + aux_Omega_b7,r6 + aux_Omega_b7,r7 + aux_Omega_b7,r8 + aux_Omega_b7,r9
        - GlobalBidder_PSI__i7,k2 - 20668026.8748422 GlobalBidder_hat{W}__i7,k2
        >= -20668026.8748422
 c3835: aux_cap_helper__b6,r9,band_LOW_PAIRED__condX1
        - aux_cap_helper__b6,r9,band_LOW_PAIRED__Z1 <= 0
 c3836: - sv_function_helpervar_b6,r2__condX2
        + 0.158453086936059 sv_function_helpervar_b6,r2__Z2 <= 0
 c3837: - 15.8453086887446 sv_function_helpervar_b2,r11__Z3
        + sv_function_helpervar_b2,r11__condX3 <= 0
 c3838: 17.2687823138999 sv_function_helpervar_b2,r5__Z2
        + 39.3192290775889 sv_function_helpervar_b2,r5__Z3
        + 5.7331288229 sv_function_helpervar_b2,r5__condX2
        + 1.7673235216 sv_function_helpervar_b2,r5__condX3
        + 114.7166938808 sv_function_helpervar_b2,r5__condX1 - aux_sv_b2,r5
         = 0
 c3839: - sv_function_helpervar_b6,r13__condX1 <= 0
 c3840: - aux_cap_helper__b9,r7,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b9,r7,band_HIGH_PAIRED__Z3 <= 0
 c3841: aux_cap_helper__b9,r3,band_LOW_PAIRED__condX2
        + aux_cap_helper__b9,r3,band_LOW_PAIRED__condX1
        - X__b9,r3,band_LOW_PAIRED  = 0
 c3842: 24.8445056357025 sv_function_helpervar_b0,r13__Z2
        + 52.0084013438652 sv_function_helpervar_b0,r13__Z3
        + 163.7425743653 sv_function_helpervar_b0,r13__condX1
        + 6.9484965889 sv_function_helpervar_b0,r13__condX2
        + 2.7822811531 sv_function_helpervar_b0,r13__condX3 - aux_sv_b0,r13
         = 0
 c3843: sv_function_helpervar_b3,r10__condX2
        - 5.86020404779476 sv_function_helpervar_b3,r10__Z2 <= 0
 c3844: X__b7,r1,band_HIGH_PAIRED + X__b0,r1,band_HIGH_PAIRED
        + X__b6,r1,band_HIGH_PAIRED + X__b3,r1,band_HIGH_PAIRED
        + X__b4,r1,band_HIGH_PAIRED + X__b5,r1,band_HIGH_PAIRED
        + X__b9,r1,band_HIGH_PAIRED + X__b2,r1,band_HIGH_PAIRED
        + X__b1,r1,band_HIGH_PAIRED + X__b8,r1,band_HIGH_PAIRED <= 3
 c3845: aux_cap_helper__b6,r1,band_UNPAIRED__Z1
        + aux_cap_helper__b6,r1,band_UNPAIRED__Z2  = 1
 c3846: sv_function_helpervar_b6,r0__condX3
        + sv_function_helpervar_b6,r0__condX2
        + sv_function_helpervar_b6,r0__condX1 - aux_c_b6,r0  = 0
 c3847: aux_cap_helper__b0,r3,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b0,r3,band_HIGH_PAIRED__Z2 <= 0
 c3848: - aux_cap_helper__b8,r4,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b8,r4,band_HIGH_PAIRED__condX1 <= 0
 c3849: aux_cap_helper__b3,r12,band_UNPAIRED__condX2
        + aux_cap_helper__b3,r12,band_UNPAIRED__condX1
        - X__b3,r12,band_UNPAIRED  = 0
 c3850: aux_cap_helper__b9,r11,band_UNPAIRED__Z2
        - aux_cap_helper__b9,r11,band_UNPAIRED__condX2 <= 0
 c3851: sv_function_helpervar_b2,r0__condX1
        - 0.158453086896187 sv_function_helpervar_b2,r0__Z1 <= 0
 c3852: aux_cap_helper__b4,r1,band_UNPAIRED__condX1
        - aux_cap_helper__b4,r1,band_UNPAIRED__Z1 <= 0
 c3853: - aux_cap_helper__b8,r8,band_UNPAIRED__condX2
        + aux_cap_helper__b8,r8,band_UNPAIRED__Z2 <= 0
 c3854: 2 aux_cap_helper__b8,r10,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b8,r10,band_HIGH_PAIRED__condX3 <= 0
 c3855: aux_cap_helper__b6,r13,band_UNPAIRED__Z2
        + aux_cap_helper__b6,r13,band_UNPAIRED__Z1  = 1
 c3856: sv_function_helpervar_b0,r1__condX1
        + sv_function_helpervar_b0,r1__condX3
        + sv_function_helpervar_b0,r1__condX2 - aux_c_b0,r1  = 0
 c3857: sv_function_helpervar_b2,r10__condX3
        + sv_function_helpervar_b2,r10__condX2
        + sv_function_helpervar_b2,r10__condX1 - aux_c_b2,r10  = 0
 c3858: sv_function_helpervar_b5,r9__condX3
        + sv_function_helpervar_b5,r9__condX2
        + sv_function_helpervar_b5,r9__condX1 - aux_c_b5,r9  = 0
 c3859: aux_cap_helper__b9,r3,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b9,r3,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b9,r3,band_HIGH_PAIRED__Z3  = 1
 c3860: - aux_cap_helper__b2,r13,band_LOW_PAIRED__Z1
        + aux_cap_helper__b2,r13,band_LOW_PAIRED__condX1 <= 0
 c3861: - aux_cap_helper__b9,r11,band_UNPAIRED__condX1 <= 0
 c3862: - aux_cap_helper__b8,r8,band_UNPAIRED__condX1 <= 0
 c3863: aux_cap_helper__b1,r12,band_UNPAIRED__condX1
        - aux_cap_helper__b1,r12,band_UNPAIRED__Z1 <= 0
 c3864: - 2 aux_cap_helper__b3,r0,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b3,r0,band_HIGH_PAIRED__condX2 <= 0
 c3865: - aux_cap_helper__b4,r1,band_LOW_PAIRED__condX2
        + aux_cap_helper__b4,r1,band_LOW_PAIRED__Z2 <= 0
 c3866: 5.69096845606432 sv_function_helpervar_b3,r6__Z3
        - sv_function_helpervar_b3,r6__condX3 <= 0
 c3867: 37.5029080882 sv_function_helpervar_b9,r7__condX3
        + 74.2283852789 sv_function_helpervar_b9,r7__condX2
        + 1995.2925271856 sv_function_helpervar_b9,r7__condX1
        + 576.718771253275 sv_function_helpervar_b9,r7__Z3
        + 304.398543408874 sv_function_helpervar_b9,r7__Z2 - aux_sv_b9,r7  = 0
 c3868: 2 aux_cap_helper__b6,r8,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b6,r8,band_HIGH_PAIRED__condX3 <= 0
 c3869: 2 aux_cap_helper__b8,r13,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b8,r13,band_HIGH_PAIRED__condX3 <= 0
 c3870: aux_cap_helper__b8,r1,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b8,r1,band_HIGH_PAIRED__Z1 <= 0
 c3871: X__b7,r13,band_HIGH_PAIRED + X__b4,r13,band_HIGH_PAIRED
        + X__b2,r13,band_HIGH_PAIRED + X__b6,r13,band_HIGH_PAIRED
        + X__b0,r13,band_HIGH_PAIRED + X__b1,r13,band_HIGH_PAIRED
        + X__b3,r13,band_HIGH_PAIRED + X__b5,r13,band_HIGH_PAIRED
        + X__b8,r13,band_HIGH_PAIRED + X__b9,r13,band_HIGH_PAIRED <= 3
 c3872: - aux_cap_helper__b3,r12,band_HIGH_PAIRED__condX1 <= 0
 c3873: GlobalBidder_PSI__i7,k2 - 20668026.8748422 GlobalBidder_hat{W}__i7,k2
        <= 0
 c3874: aux_cap_helper__b0,r9,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b0,r9,band_LOW_PAIRED__Z2 <= 0
 c3875: - aux_sv_b0,r3 + 7.17872318 sv_function_helpervar_b0,r3__condX2
        + 2.722712968 sv_function_helpervar_b0,r3__condX3
        + 163.7425744047 sv_function_helpervar_b0,r3__condX1
        + 52.9522776256028 sv_function_helpervar_b0,r3__Z3
        + 24.8080255219239 sv_function_helpervar_b0,r3__Z2  = 0
 c3876: - sv_function_helpervar_b6,r8__condX1 <= 0
 c3877: aux_cap_helper__b0,r7,band_LOW_PAIRED__condX2
        + aux_cap_helper__b0,r7,band_LOW_PAIRED__condX1
        - X__b0,r7,band_LOW_PAIRED  = 0
 c3878: - 6.52419424210838 sv_function_helpervar_b2,r9__Z2
        + sv_function_helpervar_b2,r9__condX2 <= 0
 c3879: 0.888446608 aux_cap_helper__b2,r6,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b2,r6,band_UNPAIRED__condX1
        - aux_cap_b2,r6,band_UNPAIRED
        - 0.253841888 aux_cap_helper__b2,r6,band_UNPAIRED__Z2  = 0
 c3880: - aux_cap_b5,r8,band_UNPAIRED
        + 0.888446608 aux_cap_helper__b5,r8,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b5,r8,band_UNPAIRED__condX1
        - 0.253841888 aux_cap_helper__b5,r8,band_UNPAIRED__Z2  = 0
 c3881: - X__b0,r13,band_LOW_PAIRED
        + aux_cap_helper__b0,r13,band_LOW_PAIRED__condX1
        + aux_cap_helper__b0,r13,band_LOW_PAIRED__condX2  = 0
 c3882: - sv_function_helpervar_b0,r3__condX1 <= 0
 c3883: - aux_cap_helper__b5,r8,band_UNPAIRED__condX1 <= 0
 c3884: - aux_cap_b0,r7,band_UNPAIRED
        - 0.253841888 aux_cap_helper__b0,r7,band_UNPAIRED__Z2
        + 0.888446608 aux_cap_helper__b0,r7,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b0,r7,band_UNPAIRED__condX1  = 0
 c3885: aux_cap_helper__b6,r13,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b6,r13,band_HIGH_PAIRED__Z2 <= 0
 c3886: - aux_cap_helper__b4,r4,band_LOW_PAIRED__condX2
        + aux_cap_helper__b4,r4,band_LOW_PAIRED__Z2 <= 0
 c3887: - sv_function_helpervar_b9,r5__condX2
        + 0.15845308692186 sv_function_helpervar_b9,r5__Z2 <= 0
 c3888: - aux_cap_helper__b2,r0,band_LOW_PAIRED__condX2
        + aux_cap_helper__b2,r0,band_LOW_PAIRED__Z2 <= 0
 c3889: - aux_cap_helper__b3,r4,band_UNPAIRED__condX2
        + aux_cap_helper__b3,r4,band_UNPAIRED__Z2 <= 0
 c3890: aux_cap_helper__b7,r7,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b7,r7,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b7,r7,band_HIGH_PAIRED__condX3
        - X__b7,r7,band_HIGH_PAIRED  = 0
 c3891: - aux_cap_helper__b0,r9,band_UNPAIRED__condX1 <= 0
 c3892: - aux_cap_b3,r7,band_LOW_PAIRED
        - 1.2715511236 aux_cap_helper__b3,r7,band_LOW_PAIRED__Z2
        + 4.4504289328 aux_cap_helper__b3,r7,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b3,r7,band_LOW_PAIRED__condX1  = 0
 c3893: - aux_cap_helper__b4,r3,band_UNPAIRED__condX1 <= 0
 c3894: sv_function_helpervar_b8,r10__condX1
        - 0.158453086919002 sv_function_helpervar_b8,r10__Z1 <= 0
 c3895: - 6.53397023325115 sv_function_helpervar_b0,r4__Z2
        + sv_function_helpervar_b0,r4__condX2 <= 0
 c3896: - 5.88835519733929 sv_function_helpervar_b6,r5__Z2
        + sv_function_helpervar_b6,r5__condX2 <= 0
 c3897: - aux_cap_helper__b4,r4,band_LOW_PAIRED__condX1 <= 0
 c3898: sv_function_helpervar_b4,r13__condX3
        - 15.8453086887446 sv_function_helpervar_b4,r13__Z3 <= 0
 c3899: - aux_cap_helper__b2,r0,band_LOW_PAIRED__condX1 <= 0
 c3900: aux_cap_helper__b3,r12,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b3,r12,band_HIGH_PAIRED__condX2 <= 0
 c3901: aux_cap_helper__b8,r4,band_UNPAIRED__Z2
        + aux_cap_helper__b8,r4,band_UNPAIRED__Z1  = 1
 c3902: 0.158453086900387 sv_function_helpervar_b3,r11__Z2
        - sv_function_helpervar_b3,r11__condX2 <= 0
 c3903: - X__b6,r5,band_HIGH_PAIRED
        + aux_cap_helper__b6,r5,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b6,r5,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b6,r5,band_HIGH_PAIRED__condX3  = 0
 c3904: aux_cap_helper__b1,r7,band_LOW_PAIRED__condX1
        - aux_cap_helper__b1,r7,band_LOW_PAIRED__Z1 <= 0
 c3905: - aux_cap_helper__b3,r9,band_UNPAIRED__condX1 <= 0
 c3906: aux_cap_helper__b5,r1,band_UNPAIRED__condX1
        - aux_cap_helper__b5,r1,band_UNPAIRED__Z1 <= 0
 c3907: sv_function_helpervar_b6,r4__Z2 + sv_function_helpervar_b6,r4__Z1
        + sv_function_helpervar_b6,r4__Z3  = 1
 c3908: - aux_cap_helper__b0,r4,band_UNPAIRED__condX2
        + aux_cap_helper__b0,r4,band_UNPAIRED__Z2 <= 0
 c3909: sv_function_helpervar_b0,r11__Z1 + sv_function_helpervar_b0,r11__Z2
        + sv_function_helpervar_b0,r11__Z3  = 1
 c3910: aux_cap_helper__b3,r0,band_UNPAIRED__Z2
        + aux_cap_helper__b3,r0,band_UNPAIRED__Z1  = 1
 c3911: sv_function_helpervar_b9,r10__condX1
        + sv_function_helpervar_b9,r10__condX2
        + sv_function_helpervar_b9,r10__condX3 - aux_c_b9,r10  = 0
 c3912: aux_cap_helper__b4,r5,band_LOW_PAIRED__condX1
        - aux_cap_helper__b4,r5,band_LOW_PAIRED__Z1 <= 0
 c3913: 724.221234184629 aux_sv_b0,r5 - aux_Omega_b0,r5  = 0
 c3914: aux_cap_helper__b0,r5,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b0,r5,band_HIGH_PAIRED__Z1 <= 0
 c3915: aux_cap_helper__b0,r5,band_UNPAIRED__Z2
        + aux_cap_helper__b0,r5,band_UNPAIRED__Z1  = 1
 c3916: - aux_cap_helper__b0,r2,band_LOW_PAIRED__condX1 <= 0
 c3917: - aux_Omega_b9,r0 + 142.61359382725 aux_sv_b9,r0  = 0
 c3918: 1.21690011250031 aux_cap_helper__b4,r13,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b4,r13,band_HIGH_PAIRED__Z2
        + 2.0281668543 aux_cap_helper__b4,r13,band_HIGH_PAIRED__condX1
        + 1.8253501688 aux_cap_helper__b4,r13,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b4,r13,band_HIGH_PAIRED__condX2
        - aux_cap_b4,r13,band_HIGH_PAIRED  = 0
 c3919: - sv_function_helpervar_b4,r12__condX1 <= 0
 c3920: sv_function_helpervar_b6,r9__condX2
        - 5.57476174070583 sv_function_helpervar_b6,r9__Z2 <= 0
 c3921: aux_cap_helper__b9,r0,band_UNPAIRED__condX1
        - aux_cap_helper__b9,r0,band_UNPAIRED__Z1 <= 0
 c3922: - aux_cap_helper__b2,r4,band_UNPAIRED__condX1 <= 0
 c3923: aux_cap_helper__b0,r9,band_UNPAIRED__Z2
        - aux_cap_helper__b0,r9,band_UNPAIRED__condX2 <= 0
 c3924: sv_function_helpervar_b2,r0__condX3
        - 15.8453086887446 sv_function_helpervar_b2,r0__Z3 <= 0
 c3925: aux_cap_helper__b8,r12,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b8,r12,band_LOW_PAIRED__Z2 <= 0
 c3926: - sv_function_helpervar_b5,r5__condX1 <= 0
 c3927: sv_function_helpervar_b7,r10__Z3 + sv_function_helpervar_b7,r10__Z2
        + sv_function_helpervar_b7,r10__Z1  = 1
 c3928: aux_cap_helper__b6,r7,band_UNPAIRED__condX2
        + aux_cap_helper__b6,r7,band_UNPAIRED__condX1 - X__b6,r7,band_UNPAIRED
         = 0
 c3929: sv_function_helpervar_b4,r6__condX3
        + sv_function_helpervar_b4,r6__condX1
        + sv_function_helpervar_b4,r6__condX2 - aux_c_b4,r6  = 0
 c3930: - aux_cap_helper__b0,r7,band_UNPAIRED__Z1
        + aux_cap_helper__b0,r7,band_UNPAIRED__condX1 <= 0
 c3931: aux_cap_helper__b3,r2,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b3,r2,band_HIGH_PAIRED__Z1 <= 0
 c3932: - 15.8453086887446 sv_function_helpervar_b5,r6__Z3
        + sv_function_helpervar_b5,r6__condX3 <= 0
 c3933: - X__b4,r4,band_LOW_PAIRED
        + aux_cap_helper__b4,r4,band_LOW_PAIRED__condX1
        + aux_cap_helper__b4,r4,band_LOW_PAIRED__condX2  = 0
 c3934: - 15.8453086887446 sv_function_helpervar_b8,r5__Z3
        + sv_function_helpervar_b8,r5__condX3 <= 0
 c3935: - aux_cap_helper__b7,r5,band_UNPAIRED__Z1
        + aux_cap_helper__b7,r5,band_UNPAIRED__condX1 <= 0
 c3936: X__b7,r7,band_UNPAIRED + X__b7,r7,band_LOW_PAIRED
        - GlobalBidder_W_ir__b7,r7 + X__b7,r7,band_HIGH_PAIRED >= 0
 c3937: aux_cap_helper__b4,r1,band_LOW_PAIRED__Z2
        + aux_cap_helper__b4,r1,band_LOW_PAIRED__Z1  = 1
 c3938: aux_cap_b6,r10,band_LOW_PAIRED - aux_c_b6,r10
        + aux_cap_b6,r10,band_HIGH_PAIRED + aux_cap_b6,r10,band_UNPAIRED  = 0
 c3939: 434.420234111189 aux_sv_b8,r1 - aux_Omega_b8,r1  = 0
 c3940: aux_cap_helper__b0,r2,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b0,r2,band_HIGH_PAIRED__Z1 <= 0
 c3941: 6.00433149309822 sv_function_helpervar_b4,r6__Z3
        - sv_function_helpervar_b4,r6__condX3 <= 0
 c3942: - aux_cap_b9,r11,band_HIGH_PAIRED
        + 2.8394335959 aux_cap_helper__b9,r11,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b9,r11,band_HIGH_PAIRED__condX1
        + 1.8253501688 aux_cap_helper__b9,r11,band_HIGH_PAIRED__condX3
        - 0.8112667416 aux_cap_helper__b9,r11,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b9,r11,band_HIGH_PAIRED__Z3  = 0
 c3943: 6.29731842908851 sv_function_helpervar_b0,r8__Z3
        - sv_function_helpervar_b0,r8__condX3 <= 0
 c3944: - sv_function_helpervar_b1,r2__condX3
        + 5.66692189745523 sv_function_helpervar_b1,r2__Z3 <= 0
 c3945: - 2 aux_cap_helper__b2,r1,band_UNPAIRED__Z2
        + aux_cap_helper__b2,r1,band_UNPAIRED__condX2 <= 0
 c3946: - aux_cap_helper__b3,r9,band_UNPAIRED__condX2
        + aux_cap_helper__b3,r9,band_UNPAIRED__Z2 <= 0
 c3947: - aux_cap_helper__b6,r0,band_UNPAIRED__Z1
        + aux_cap_helper__b6,r0,band_UNPAIRED__condX1 <= 0
 c3948: - aux_cap_helper__b6,r3,band_LOW_PAIRED__Z1
        + aux_cap_helper__b6,r3,band_LOW_PAIRED__condX1 <= 0
 c3949: aux_cap_helper__b6,r10,band_LOW_PAIRED__condX1
        + aux_cap_helper__b6,r10,band_LOW_PAIRED__condX2
        - X__b6,r10,band_LOW_PAIRED  = 0
 c3950: - sv_function_helpervar_b5,r5__condX2
        + 0.158453086904847 sv_function_helpervar_b5,r5__Z2 <= 0
 c3951: aux_cap_helper__b0,r2,band_LOW_PAIRED__Z2
        - aux_cap_helper__b0,r2,band_LOW_PAIRED__condX2 <= 0
 c3952: - X__b0,r9,band_HIGH_PAIRED
        + aux_cap_helper__b0,r9,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b0,r9,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b0,r9,band_HIGH_PAIRED__condX1  = 0
 c3953: aux_cap_helper__b5,r13,band_UNPAIRED__condX1
        - aux_cap_helper__b5,r13,band_UNPAIRED__Z1 <= 0
 c3954: aux_cap_helper__b2,r3,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b2,r3,band_HIGH_PAIRED__Z3 <= 0
 c3955: aux_cap_helper__b9,r5,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b9,r5,band_HIGH_PAIRED__condX2 <= 0
 c3956: aux_cap_helper__b7,r5,band_UNPAIRED__Z2
        + aux_cap_helper__b7,r5,band_UNPAIRED__Z1  = 1
 c3957: 0.158453086891435 sv_function_helpervar_b6,r10__Z2
        - sv_function_helpervar_b6,r10__condX2 <= 0
 c3958: - aux_c_b2,r1 + sv_function_helpervar_b2,r1__condX1
        + sv_function_helpervar_b2,r1__condX3
        + sv_function_helpervar_b2,r1__condX2  = 0
 c3959: - aux_cap_b5,r2,band_LOW_PAIRED
        + 4.4504289328 aux_cap_helper__b5,r2,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b5,r2,band_LOW_PAIRED__condX1
        - 1.2715511236 aux_cap_helper__b5,r2,band_LOW_PAIRED__Z2  = 0
 c3960: aux_cap_helper__b5,r5,band_LOW_PAIRED__Z1
        + aux_cap_helper__b5,r5,band_LOW_PAIRED__Z2  = 1
 c3961: aux_cap_helper__b2,r11,band_UNPAIRED__Z1
        + aux_cap_helper__b2,r11,band_UNPAIRED__Z2  = 1
 c3962: - X__b3,r9,band_HIGH_PAIRED
        + aux_cap_helper__b3,r9,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b3,r9,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b3,r9,band_HIGH_PAIRED__condX1  = 0
 c3963: aux_cap_helper__b0,r1,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b0,r1,band_LOW_PAIRED__Z2 <= 0
 c3964: - aux_cap_helper__b4,r1,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b4,r1,band_HIGH_PAIRED__Z3 <= 0
 c3965: - aux_Omega_b3,r2 + 197.167219534338 aux_sv_b3,r2  = 0
 c3966: aux_cap_helper__b4,r8,band_LOW_PAIRED__Z1
        + aux_cap_helper__b4,r8,band_LOW_PAIRED__Z2  = 1
 c3967: 1995.2925271012 sv_function_helpervar_b9,r0__condX1
        + 87.8448485885 sv_function_helpervar_b9,r0__condX2
        + 33.0881267142 sv_function_helpervar_b9,r0__condX3
        + 302.240972764288 sv_function_helpervar_b9,r0__Z2
        + 646.672344917651 sv_function_helpervar_b9,r0__Z3 - aux_sv_b9,r0  = 0
 c3968: 0.158453086963894 sv_function_helpervar_b1,r13__Z2
        - sv_function_helpervar_b1,r13__condX2 <= 0
 c3969: - aux_cap_helper__b2,r10,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b2,r10,band_HIGH_PAIRED__Z3 <= 0
 c3970: - aux_cap_helper__b9,r0,band_LOW_PAIRED__condX1 <= 0
 c3971: aux_cap_helper__b9,r13,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b9,r13,band_UNPAIRED__Z2 <= 0
 c3972: sv_function_helpervar_b9,r0__condX1
        + sv_function_helpervar_b9,r0__condX2
        + sv_function_helpervar_b9,r0__condX3 - aux_c_b9,r0  = 0
 c3973: aux_cap_helper__b4,r4,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b4,r4,band_HIGH_PAIRED__condX2 <= 0
 c3974: - aux_Omega_b0,r10 + 596.617850455376 aux_sv_b0,r10  = 0
 c3975: 2 aux_cap_helper__b2,r0,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b2,r0,band_HIGH_PAIRED__condX3 <= 0
 c3976: 2 aux_cap_helper__b9,r8,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b9,r8,band_HIGH_PAIRED__condX3 <= 0
 c3977: - 14 GlobalBidder_hat{W}__i8,k1 + GlobalBidder_W_i_8 >= -1
 c3978: - aux_sv_b4,r10 + 132.838773668131 sv_function_helpervar_b4,r10__Z2
        + 305.040238585568 sv_function_helpervar_b4,r10__Z3
        + 883.4028074445 sv_function_helpervar_b4,r10__condX1
        + 13.4674836044 sv_function_helpervar_b4,r10__condX3
        + 45.0551537353 sv_function_helpervar_b4,r10__condX2  = 0
 c3979: aux_cap_helper__b1,r5,band_LOW_PAIRED__Z1
        + aux_cap_helper__b1,r5,band_LOW_PAIRED__Z2  = 1
 c3980: - aux_cap_helper__b4,r4,band_HIGH_PAIRED__condX1 <= 0
 c3981: aux_cap_helper__b7,r2,band_UNPAIRED__condX1
        + aux_cap_helper__b7,r2,band_UNPAIRED__condX2 - X__b7,r2,band_UNPAIRED
         = 0
 c3982: - 2 aux_cap_helper__b7,r9,band_LOW_PAIRED__Z2
        + aux_cap_helper__b7,r9,band_LOW_PAIRED__condX2 <= 0
 c3983: aux_cap_helper__b1,r13,band_UNPAIRED__Z2
        + aux_cap_helper__b1,r13,band_UNPAIRED__Z1  = 1
 c3984: sv_function_helpervar_b4,r7__Z1 + sv_function_helpervar_b4,r7__Z3
        + sv_function_helpervar_b4,r7__Z2  = 1
 c3985: sv_function_helpervar_b9,r11__Z1 + sv_function_helpervar_b9,r11__Z2
        + sv_function_helpervar_b9,r11__Z3  = 1
 c3986: aux_cap_b0,r0,band_LOW_PAIRED + aux_cap_b0,r0,band_UNPAIRED
        - aux_c_b0,r0 + aux_cap_b0,r0,band_HIGH_PAIRED  = 0
 c3987: - sv_function_helpervar_b1,r11__condX2
        + 0.1584530868979 sv_function_helpervar_b1,r11__Z2 <= 0
 c3988: 0.0102040816326531 X__b7,r9,band_LOW_PAIRED - GlobalBidder_W_ir__b7,r9
        + 0.0102040816326531 X__b7,r9,band_UNPAIRED
        + 0.0102040816326531 X__b7,r9,band_HIGH_PAIRED <= 0
 c3989: aux_cap_helper__b0,r10,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b0,r10,band_HIGH_PAIRED__Z2 <= 0
 c3990: - aux_cap_b4,r1,band_UNPAIRED
        + 0.888446608 aux_cap_helper__b4,r1,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b4,r1,band_UNPAIRED__condX1
        - 0.253841888 aux_cap_helper__b4,r1,band_UNPAIRED__Z2  = 0
 c3991: aux_cap_helper__b7,r13,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b7,r13,band_LOW_PAIRED__Z2 <= 0
 c3992: - X__b0,r9,band_LOW_PAIRED
        + aux_cap_helper__b0,r9,band_LOW_PAIRED__condX1
        + aux_cap_helper__b0,r9,band_LOW_PAIRED__condX2  = 0
 c3993: - 15.8453086887446 sv_function_helpervar_b4,r12__Z3
        + sv_function_helpervar_b4,r12__condX3 <= 0
 c3994: aux_cap_b5,r2,band_LOW_PAIRED + aux_cap_b5,r2,band_HIGH_PAIRED
        + aux_cap_b5,r2,band_UNPAIRED - aux_c_b5,r2  = 0
 c3995: aux_cap_helper__b7,r1,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b7,r1,band_LOW_PAIRED__Z2 <= 0
 c3996: 2 aux_cap_helper__b7,r1,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b7,r1,band_HIGH_PAIRED__condX3 <= 0
 c3997: aux_cap_helper__b0,r4,band_UNPAIRED__Z1
        + aux_cap_helper__b0,r4,band_UNPAIRED__Z2  = 1
 c3998: aux_cap_helper__b9,r0,band_LOW_PAIRED__Z2
        - aux_cap_helper__b9,r0,band_LOW_PAIRED__condX2 <= 0
 c3999: - aux_cap_helper__b9,r5,band_HIGH_PAIRED__condX1 <= 0
 c4000: aux_cap_helper__b3,r10,band_UNPAIRED__condX1
        - aux_cap_helper__b3,r10,band_UNPAIRED__Z1 <= 0
 c4001: aux_cap_helper__b5,r6,band_UNPAIRED__condX1
        - aux_cap_helper__b5,r6,band_UNPAIRED__Z1 <= 0
 c4002: - aux_sv_b0,r4 + 24.8469111139546 sv_function_helpervar_b0,r4__Z2
        + 51.9424513231026 sv_function_helpervar_b0,r4__Z3
        + 2.7864432696 sv_function_helpervar_b0,r4__condX3
        + 6.9333155757 sv_function_helpervar_b0,r4__condX2
        + 163.7425743442 sv_function_helpervar_b0,r4__condX1  = 0
 c4003: aux_cap_helper__b4,r3,band_UNPAIRED__Z2
        - aux_cap_helper__b4,r3,band_UNPAIRED__condX2 <= 0
 c4004: aux_cap_helper__b8,r1,band_LOW_PAIRED__condX2
        + aux_cap_helper__b8,r1,band_LOW_PAIRED__condX1
        - X__b8,r1,band_LOW_PAIRED  = 0
 c4005: - 2 aux_cap_helper__b5,r9,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b5,r9,band_HIGH_PAIRED__condX2 <= 0
 c4006: sv_function_helpervar_b1,r9__condX2
        - 5.92566887890375 sv_function_helpervar_b1,r9__Z2 <= 0
 c4007: aux_cap_helper__b2,r6,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b2,r6,band_HIGH_PAIRED__Z3 <= 0
 c4008: aux_cap_helper__b0,r13,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b0,r13,band_HIGH_PAIRED__Z2 <= 0
 c4009: aux_cap_helper__b0,r6,band_UNPAIRED__condX2
        + aux_cap_helper__b0,r6,band_UNPAIRED__condX1 - X__b0,r6,band_UNPAIRED
         = 0
 c4010: aux_cap_helper__b5,r12,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b5,r12,band_HIGH_PAIRED__Z2 <= 0
 c4011: aux_cap_helper__b2,r9,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b2,r9,band_HIGH_PAIRED__Z2 <= 0
 c4012: 0.15845308689699 sv_function_helpervar_b2,r12__Z2
        - sv_function_helpervar_b2,r12__condX2 <= 0
 c4013: aux_cap_helper__b0,r2,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b0,r2,band_UNPAIRED__Z2 <= 0
 c4014: - 1.2715511236 aux_cap_helper__b2,r4,band_LOW_PAIRED__Z2
        + 3.1788778092 aux_cap_helper__b2,r4,band_LOW_PAIRED__condX1
        + 4.4504289328 aux_cap_helper__b2,r4,band_LOW_PAIRED__condX2
        - aux_cap_b2,r4,band_LOW_PAIRED  = 0
 c4015: - aux_cap_helper__b8,r7,band_LOW_PAIRED__Z1
        + aux_cap_helper__b8,r7,band_LOW_PAIRED__condX1 <= 0
 c4016: aux_cap_b8,r1,band_HIGH_PAIRED + aux_cap_b8,r1,band_LOW_PAIRED
        - aux_c_b8,r1 + aux_cap_b8,r1,band_UNPAIRED  = 0
 c4017: - X__b0,r13,band_UNPAIRED
        + aux_cap_helper__b0,r13,band_UNPAIRED__condX2
        + aux_cap_helper__b0,r13,band_UNPAIRED__condX1  = 0
 c4018: 36.2080029138384 aux_sv_b9,r13 - aux_Omega_b9,r13  = 0
 c4019: - 3 aux_cap_helper__b2,r13,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b2,r13,band_HIGH_PAIRED__condX3 <= 0
 c4020: - aux_Omega_b7,r3 + 674.756622976447 aux_sv_b7,r3  = 0
 c4021: aux_cap_helper__b7,r12,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b7,r12,band_HIGH_PAIRED__Z3 <= 0
 c4022: - 3 aux_cap_helper__b5,r3,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b5,r3,band_HIGH_PAIRED__condX3 <= 0
 c4023: aux_cap_helper__b2,r6,band_UNPAIRED__condX1
        - aux_cap_helper__b2,r6,band_UNPAIRED__Z1 <= 0
 c4024: aux_cap_helper__b8,r8,band_LOW_PAIRED__Z1
        + aux_cap_helper__b8,r8,band_LOW_PAIRED__Z2  = 1
 c4025: - 2 aux_cap_helper__b5,r6,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b5,r6,band_HIGH_PAIRED__condX2 <= 0
 c4026: - 20668026.8748422 GlobalBidder_hat{W}__i8,k0 + GlobalBidder_PSI__i8,k0
        <= 0
 c4027: X__b0,r9,band_LOW_PAIRED + X__b9,r9,band_LOW_PAIRED
        + X__b2,r9,band_LOW_PAIRED + X__b6,r9,band_LOW_PAIRED
        + X__b7,r9,band_LOW_PAIRED + X__b5,r9,band_LOW_PAIRED
        + X__b8,r9,band_LOW_PAIRED + X__b4,r9,band_LOW_PAIRED
        + X__b1,r9,band_LOW_PAIRED + X__b3,r9,band_LOW_PAIRED <= 2
 c4028: - aux_cap_helper__b6,r6,band_HIGH_PAIRED__condX1 <= 0
 c4029: aux_cap_helper__b7,r12,band_UNPAIRED__Z2
        + aux_cap_helper__b7,r12,band_UNPAIRED__Z1  = 1
 c4030: X__b8,r11,band_LOW_PAIRED + X__b8,r11,band_HIGH_PAIRED
        - GlobalBidder_W_ir__b8,r11 + X__b8,r11,band_UNPAIRED >= 0
 c4031: aux_cap_helper__b2,r7,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b2,r7,band_HIGH_PAIRED__Z3 <= 0
 c4032: sv_function_helpervar_b2,r1__condX3
        - 15.8453086887446 sv_function_helpervar_b2,r1__Z3 <= 0
 c4033: 5.6968505612505 sv_function_helpervar_b2,r7__Z3
        - sv_function_helpervar_b2,r7__condX3 <= 0
 c4034: - 0.253841888 aux_cap_helper__b7,r5,band_UNPAIRED__Z2
        - aux_cap_b7,r5,band_UNPAIRED
        + 0.63460472 aux_cap_helper__b7,r5,band_UNPAIRED__condX1
        + 0.888446608 aux_cap_helper__b7,r5,band_UNPAIRED__condX2  = 0
 c4035: - sv_function_helpervar_b5,r4__condX1 <= 0
 c4036: - aux_c_b7,r9 + sv_function_helpervar_b7,r9__condX1
        + sv_function_helpervar_b7,r9__condX2
        + sv_function_helpervar_b7,r9__condX3  = 0
 c4037: - aux_cap_b4,r3,band_UNPAIRED
        - 0.253841888 aux_cap_helper__b4,r3,band_UNPAIRED__Z2
        + 0.63460472 aux_cap_helper__b4,r3,band_UNPAIRED__condX1
        + 0.888446608 aux_cap_helper__b4,r3,band_UNPAIRED__condX2  = 0
 c4038: aux_cap_helper__b4,r7,band_LOW_PAIRED__Z2
        - aux_cap_helper__b4,r7,band_LOW_PAIRED__condX2 <= 0
 c4039: - aux_cap_helper__b4,r12,band_UNPAIRED__condX1 <= 0
 c4040: sv_function_helpervar_b8,r6__condX3
        - 15.8453086887446 sv_function_helpervar_b8,r6__Z3 <= 0
 c4041: - X__b9,r1,band_LOW_PAIRED
        + aux_cap_helper__b9,r1,band_LOW_PAIRED__condX2
        + aux_cap_helper__b9,r1,band_LOW_PAIRED__condX1  = 0
 c4042: 5.43751382574535 sv_function_helpervar_b3,r5__Z3
        - sv_function_helpervar_b3,r5__condX3 <= 0
 c4043: aux_cap_helper__b8,r10,band_UNPAIRED__Z2
        + aux_cap_helper__b8,r10,band_UNPAIRED__Z1  = 1
 c4044: - 0.253841888 aux_cap_helper__b5,r0,band_UNPAIRED__Z2
        - aux_cap_b5,r0,band_UNPAIRED
        + 0.888446608 aux_cap_helper__b5,r0,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b5,r0,band_UNPAIRED__condX1  = 0
 c4045: - 0.253841888 aux_cap_helper__b8,r2,band_UNPAIRED__Z2
        - aux_cap_b8,r2,band_UNPAIRED
        + 0.888446608 aux_cap_helper__b8,r2,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b8,r2,band_UNPAIRED__condX1  = 0
 c4046: - aux_cap_helper__b7,r9,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b7,r9,band_HIGH_PAIRED__condX1 <= 0
 c4047: - 6.80737543489106 sv_function_helpervar_b7,r9__Z2
        + sv_function_helpervar_b7,r9__condX2 <= 0
 c4048: - X__b2,r10,band_LOW_PAIRED
        + aux_cap_helper__b2,r10,band_LOW_PAIRED__condX2
        + aux_cap_helper__b2,r10,band_LOW_PAIRED__condX1  = 0
 c4049: - aux_c_b2,r9 + sv_function_helpervar_b2,r9__condX1
        + sv_function_helpervar_b2,r9__condX2
        + sv_function_helpervar_b2,r9__condX3  = 0
 c4050: sv_function_helpervar_b7,r12__condX1
        - 0.158453086918105 sv_function_helpervar_b7,r12__Z1 <= 0
 c4051: aux_cap_helper__b0,r7,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b0,r7,band_HIGH_PAIRED__Z2 <= 0
 c4052: - sv_function_helpervar_b4,r6__condX1 <= 0
 c4053: aux_cap_helper__b3,r5,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b3,r5,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b3,r5,band_HIGH_PAIRED__Z1  = 1
 c4054: sv_function_helpervar_b8,r6__condX2
        + sv_function_helpervar_b8,r6__condX1
        + sv_function_helpervar_b8,r6__condX3 - aux_c_b8,r6  = 0
 c4055: - X__b1,r8,band_HIGH_PAIRED
        + aux_cap_helper__b1,r8,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b1,r8,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b1,r8,band_HIGH_PAIRED__condX3  = 0
 c4056: - aux_cap_helper__b2,r12,band_HIGH_PAIRED__condX1 <= 0
 c4057: aux_cap_helper__b4,r4,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b4,r4,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b4,r4,band_HIGH_PAIRED__Z2  = 1
 c4058: - 2 aux_cap_helper__b5,r0,band_UNPAIRED__Z2
        + aux_cap_helper__b5,r0,band_UNPAIRED__condX2 <= 0
 c4059: aux_cap_helper__b1,r13,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b1,r13,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b1,r13,band_HIGH_PAIRED__condX3
        - X__b1,r13,band_HIGH_PAIRED  = 0
 c4060: - sv_function_helpervar_b7,r5__condX3
        + 8.07903836345579 sv_function_helpervar_b7,r5__Z3 <= 0
 c4061: 0.158453086894059 sv_function_helpervar_b1,r5__Z2
        - sv_function_helpervar_b1,r5__condX2 <= 0
 c4062: - aux_cap_helper__b8,r3,band_LOW_PAIRED__condX1 <= 0
 c4063: - aux_cap_helper__b6,r6,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b6,r6,band_HIGH_PAIRED__Z2 <= 0
 c4064: aux_cap_helper__b1,r4,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b1,r4,band_HIGH_PAIRED__condX2 <= 0
 c4065: 70.2256035106 sv_function_helpervar_b9,r3__condX2
        + 1995.2925271234 sv_function_helpervar_b9,r3__condX1
        + 39.4378534283 sv_function_helpervar_b9,r3__condX3 - aux_sv_b9,r3
        + 546.058965043206 sv_function_helpervar_b9,r3__Z3
        + 305.032796535893 sv_function_helpervar_b9,r3__Z2  = 0
 c4066: 0.68832 GlobalBidder_PSI__i8,k2 + 0.92208 GlobalBidder_PSI__i8,k1
        + GlobalBidder_PSI__i8,k0 - v__8  = 0
 c4067: aux_cap_helper__b1,r13,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b1,r13,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b1,r13,band_HIGH_PAIRED__Z3  = 1
 c4068: - aux_cap_helper__b9,r2,band_HIGH_PAIRED__condX1 <= 0
 c4069: - X__b6,r9,band_HIGH_PAIRED
        + aux_cap_helper__b6,r9,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b6,r9,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b6,r9,band_HIGH_PAIRED__condX1  = 0
 c4070: aux_cap_helper__b1,r2,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b1,r2,band_UNPAIRED__Z2 <= 0
 c4071: - aux_cap_helper__b9,r12,band_HIGH_PAIRED__condX1 <= 0
 c4072: - sv_function_helpervar_b6,r3__condX3
        + 6.01419287738594 sv_function_helpervar_b6,r3__Z3 <= 0
 c4073: - aux_cap_helper__b1,r4,band_HIGH_PAIRED__condX1 <= 0
 c4074: sv_function_helpervar_b3,r6__Z1 + sv_function_helpervar_b3,r6__Z3
        + sv_function_helpervar_b3,r6__Z2  = 1
 c4075: - aux_cap_helper__b3,r12,band_UNPAIRED__condX2
        + aux_cap_helper__b3,r12,band_UNPAIRED__Z2 <= 0
 c4076: - aux_cap_helper__b7,r3,band_HIGH_PAIRED__condX1 <= 0
 c4077: aux_cap_helper__b9,r10,band_LOW_PAIRED__Z2
        + aux_cap_helper__b9,r10,band_LOW_PAIRED__Z1  = 1
 c4078: aux_cap_helper__b0,r1,band_LOW_PAIRED__Z1
        + aux_cap_helper__b0,r1,band_LOW_PAIRED__Z2  = 1
 c4079: sv_function_helpervar_b0,r5__condX3
        - 15.8453086887446 sv_function_helpervar_b0,r5__Z3 <= 0
 c4080: aux_cap_helper__b2,r12,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b2,r12,band_HIGH_PAIRED__condX2 <= 0
 c4081: - sv_function_helpervar_b2,r11__condX1 <= 0
 c4082: aux_cap_helper__b6,r11,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b6,r11,band_HIGH_PAIRED__Z1 <= 0
 c4083: aux_cap_helper__b1,r6,band_UNPAIRED__condX1
        - aux_cap_helper__b1,r6,band_UNPAIRED__Z1 <= 0
 c4084: - aux_cap_helper__b7,r2,band_UNPAIRED__condX1 <= 0
 c4085: - aux_cap_helper__b4,r7,band_LOW_PAIRED__condX1 <= 0
 c4086: - aux_cap_helper__b8,r3,band_LOW_PAIRED__condX2
        + aux_cap_helper__b8,r3,band_LOW_PAIRED__Z2 <= 0
 c4087: - 2 aux_cap_helper__b9,r7,band_LOW_PAIRED__Z2
        + aux_cap_helper__b9,r7,band_LOW_PAIRED__condX2 <= 0
 c4088: aux_cap_helper__b9,r7,band_UNPAIRED__Z1
        + aux_cap_helper__b9,r7,band_UNPAIRED__Z2  = 1
 c4089: aux_cap_helper__b8,r0,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b8,r0,band_HIGH_PAIRED__Z1 <= 0
 c4090: - sv_function_helpervar_b0,r4__condX1 <= 0
 c4091: - aux_cap_helper__b9,r12,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b9,r12,band_HIGH_PAIRED__Z2 <= 0
 c4092: - 15.8453086887446 sv_function_helpervar_b2,r12__Z3
        + sv_function_helpervar_b2,r12__condX3 <= 0
 c4093: - X__b3,r2,band_LOW_PAIRED
        + aux_cap_helper__b3,r2,band_LOW_PAIRED__condX2
        + aux_cap_helper__b3,r2,band_LOW_PAIRED__condX1  = 0
 c4094: 0.0102040816326531 X__b8,r9,band_LOW_PAIRED
        + 0.0102040816326531 X__b8,r9,band_HIGH_PAIRED
        + 0.0102040816326531 X__b8,r9,band_UNPAIRED - GlobalBidder_W_ir__b8,r9
        <= 0
 c4095: - 0.158453086913542 sv_function_helpervar_b4,r12__Z1
        + sv_function_helpervar_b4,r12__condX1 <= 0
 c4096: aux_cap_helper__b8,r3,band_UNPAIRED__Z1
        + aux_cap_helper__b8,r3,band_UNPAIRED__Z2  = 1
 c4097: aux_cap_helper__b4,r12,band_UNPAIRED__Z2
        - aux_cap_helper__b4,r12,band_UNPAIRED__condX2 <= 0
 c4098: aux_cap_helper__b2,r12,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b2,r12,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b2,r12,band_HIGH_PAIRED__Z1  = 1
 c4099: aux_cap_helper__b7,r3,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b7,r3,band_HIGH_PAIRED__condX2 <= 0
 c4100: - aux_c_b8,r7 + aux_cap_b8,r7,band_HIGH_PAIRED
        + aux_cap_b8,r7,band_UNPAIRED + aux_cap_b8,r7,band_LOW_PAIRED  = 0
 c4101: - 2 aux_cap_helper__b0,r0,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b0,r0,band_HIGH_PAIRED__condX2 <= 0
 c4102: sv_function_helpervar_b4,r4__Z1 + sv_function_helpervar_b4,r4__Z2
        + sv_function_helpervar_b4,r4__Z3  = 1
 c4103: aux_cap_helper__b4,r0,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b4,r0,band_UNPAIRED__Z2 <= 0
 c4104: aux_cap_helper__b0,r6,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b0,r6,band_UNPAIRED__Z2 <= 0
 c4105: - X__b6,r1,band_LOW_PAIRED
        + aux_cap_helper__b6,r1,band_LOW_PAIRED__condX2
        + aux_cap_helper__b6,r1,band_LOW_PAIRED__condX1  = 0
 c4106: - X__b1,r4,band_LOW_PAIRED
        + aux_cap_helper__b1,r4,band_LOW_PAIRED__condX2
        + aux_cap_helper__b1,r4,band_LOW_PAIRED__condX1  = 0
 c4107: - 2 aux_cap_helper__b8,r5,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b8,r5,band_HIGH_PAIRED__condX2 <= 0
 c4108: aux_cap_helper__b2,r2,band_UNPAIRED__condX1
        - aux_cap_helper__b2,r2,band_UNPAIRED__Z1 <= 0
 c4109: aux_cap_helper__b6,r9,band_UNPAIRED__Z2
        - aux_cap_helper__b6,r9,band_UNPAIRED__condX2 <= 0
 c4110: - 0.158453086934078 sv_function_helpervar_b5,r2__Z1
        + sv_function_helpervar_b5,r2__condX1 <= 0
 c4111: - aux_cap_helper__b3,r12,band_UNPAIRED__condX1 <= 0
 c4112: - aux_cap_helper__b7,r11,band_UNPAIRED__Z1
        + aux_cap_helper__b7,r11,band_UNPAIRED__condX1 <= 0
 c4113: 1.21690011250031 aux_cap_helper__b5,r11,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b5,r11,band_HIGH_PAIRED__Z2
        - aux_cap_b5,r11,band_HIGH_PAIRED
        + 1.8253501688 aux_cap_helper__b5,r11,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b5,r11,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b5,r11,band_HIGH_PAIRED__condX1  = 0
 c4114: 3.1788778092 aux_cap_helper__b5,r7,band_LOW_PAIRED__condX1
        + 4.4504289328 aux_cap_helper__b5,r7,band_LOW_PAIRED__condX2
        - 1.2715511236 aux_cap_helper__b5,r7,band_LOW_PAIRED__Z2
        - aux_cap_b5,r7,band_LOW_PAIRED  = 0
 c4115: 2 aux_cap_helper__b6,r7,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b6,r7,band_HIGH_PAIRED__condX3 <= 0
 c4116: 629.226997157066 sv_function_helpervar_b9,r1__Z3
        + 302.906083552658 sv_function_helpervar_b9,r1__Z2 - aux_sv_b9,r1
        + 1995.2925266659 sv_function_helpervar_b9,r1__condX1
        + 83.6473236033 sv_function_helpervar_b9,r1__condX2
        + 34.1891054394 sv_function_helpervar_b9,r1__condX3  = 0
 c4117: aux_cap_helper__b8,r6,band_LOW_PAIRED__Z2
        + aux_cap_helper__b8,r6,band_LOW_PAIRED__Z1  = 1
 c4118: sv_function_helpervar_b1,r13__Z1 + sv_function_helpervar_b1,r13__Z3
        + sv_function_helpervar_b1,r13__Z2  = 1
 c4119: aux_cap_helper__b4,r6,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b4,r6,band_HIGH_PAIRED__Z3 <= 0
 c4120: 2 aux_cap_helper__b7,r2,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b7,r2,band_HIGH_PAIRED__condX3 <= 0
 c4121: 684.151486680195 aux_sv_b1,r12 - aux_Omega_b1,r12  = 0
 c4122: aux_cap_b4,r9,band_UNPAIRED + aux_cap_b4,r9,band_LOW_PAIRED
        + aux_cap_b4,r9,band_HIGH_PAIRED - aux_c_b4,r9  = 0
 c4123: - aux_cap_helper__b3,r4,band_UNPAIRED__condX1 <= 0
 c4124: - X__b9,r10,band_UNPAIRED
        + aux_cap_helper__b9,r10,band_UNPAIRED__condX2
        + aux_cap_helper__b9,r10,band_UNPAIRED__condX1  = 0
 c4125: aux_cap_helper__b0,r6,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b0,r6,band_HIGH_PAIRED__Z2 <= 0
 c4126: aux_cap_helper__b3,r11,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b3,r11,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b3,r11,band_HIGH_PAIRED__Z3  = 1
 c4127: aux_cap_helper__b5,r3,band_LOW_PAIRED__Z2
        + aux_cap_helper__b5,r3,band_LOW_PAIRED__Z1  = 1
 c4128: 229.854863894583 aux_sv_b8,r2 - aux_Omega_b8,r2  = 0
 c4129: sv_function_helpervar_b5,r5__condX3
        - 15.8453086887446 sv_function_helpervar_b5,r5__Z3 <= 0
 c4130: 0.158453086934238 sv_function_helpervar_b0,r1__Z2
        - sv_function_helpervar_b0,r1__condX2 <= 0
 c4131: - aux_cap_helper__b2,r11,band_HIGH_PAIRED__condX1 <= 0
 c4132: - aux_cap_helper__b2,r6,band_LOW_PAIRED__condX1 <= 0
 c4133: - 0.253841888 aux_cap_helper__b7,r3,band_UNPAIRED__Z2
        - aux_cap_b7,r3,band_UNPAIRED
        + 0.63460472 aux_cap_helper__b7,r3,band_UNPAIRED__condX1
        + 0.888446608 aux_cap_helper__b7,r3,band_UNPAIRED__condX2  = 0
 c4134: - X__b3,r9,band_LOW_PAIRED
        + aux_cap_helper__b3,r9,band_LOW_PAIRED__condX2
        + aux_cap_helper__b3,r9,band_LOW_PAIRED__condX1  = 0
 c4135: - 2 aux_cap_helper__b7,r6,band_LOW_PAIRED__Z2
        + aux_cap_helper__b7,r6,band_LOW_PAIRED__condX2 <= 0
 c4136: sv_function_helpervar_b3,r3__condX2
        - 5.74780324936675 sv_function_helpervar_b3,r3__Z2 <= 0
 c4137: 147.615314417963 sv_function_helpervar_b5,r3__Z2
        + 331.134535001395 sv_function_helpervar_b5,r3__Z3 - aux_sv_b5,r3
        + 15.3576707479 sv_function_helpervar_b5,r3__condX3
        + 978.9018726958 sv_function_helpervar_b5,r3__condX1
        + 47.2992308697 sv_function_helpervar_b5,r3__condX2  = 0
 c4138: aux_cap_helper__b2,r11,band_LOW_PAIRED__Z2
        + aux_cap_helper__b2,r11,band_LOW_PAIRED__Z1  = 1
 c4139: - sv_function_helpervar_b0,r10__condX1 <= 0
 c4140: aux_cap_helper__b3,r8,band_LOW_PAIRED__condX1
        - aux_cap_helper__b3,r8,band_LOW_PAIRED__Z1 <= 0
 c4141: sv_function_helpervar_b8,r7__condX2
        - 6.83798725563921 sv_function_helpervar_b8,r7__Z2 <= 0
 c4142: - X__b9,r5,band_HIGH_PAIRED
        + aux_cap_helper__b9,r5,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b9,r5,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b9,r5,band_HIGH_PAIRED__condX2  = 0
 c4143: sv_function_helpervar_b7,r11__condX1
        - 0.158453086948001 sv_function_helpervar_b7,r11__Z1 <= 0
 c4144: - sv_function_helpervar_b3,r13__condX2
        + 0.158453086929357 sv_function_helpervar_b3,r13__Z2 <= 0
 c4145: - aux_cap_helper__b1,r9,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b1,r9,band_HIGH_PAIRED__Z3 <= 0
 c4146: aux_cap_helper__b7,r10,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b7,r10,band_LOW_PAIRED__Z2 <= 0
 c4147: aux_cap_helper__b7,r6,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b7,r6,band_UNPAIRED__Z2 <= 0
 c4148: sv_function_helpervar_b9,r10__condX3
        - 15.8453086887446 sv_function_helpervar_b9,r10__Z3 <= 0
 c4149: - X__b7,r4,band_HIGH_PAIRED
        + aux_cap_helper__b7,r4,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b7,r4,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b7,r4,band_HIGH_PAIRED__condX3  = 0
 c4150: 7.4860969108 sv_function_helpervar_b1,r7__condX2
        + 150.9476061308 sv_function_helpervar_b1,r7__condX1
        + 2.3349519701 sv_function_helpervar_b1,r7__condX3 - aux_sv_b1,r7
        + 22.7319189905248 sv_function_helpervar_b1,r7__Z2
        + 51.5875732423233 sv_function_helpervar_b1,r7__Z3  = 0
 c4151: sv_function_helpervar_b0,r5__condX2
        - 6.46449030410519 sv_function_helpervar_b0,r5__Z2 <= 0
 c4152: aux_cap_helper__b6,r0,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b6,r0,band_HIGH_PAIRED__Z2 <= 0
 c4153: 0.0102040816326531 X__b9,r0,band_HIGH_PAIRED
        + 0.0102040816326531 X__b9,r0,band_LOW_PAIRED
        - GlobalBidder_W_ir__b9,r0 + 0.0102040816326531 X__b9,r0,band_UNPAIRED
        <= 0
 c4154: - aux_cap_helper__b5,r4,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b5,r4,band_HIGH_PAIRED__condX1 <= 0
 c4155: - aux_Omega_b6,r1 + 155.092610662511 aux_sv_b6,r1  = 0
 c4156: 6.24539851566527 sv_function_helpervar_b4,r7__Z3
        - sv_function_helpervar_b4,r7__condX3 <= 0
 c4157: - aux_c_b7,r10 + aux_cap_b7,r10,band_HIGH_PAIRED
        + aux_cap_b7,r10,band_UNPAIRED + aux_cap_b7,r10,band_LOW_PAIRED  = 0
 c4158: - aux_cap_helper__b2,r6,band_LOW_PAIRED__condX2
        + aux_cap_helper__b2,r6,band_LOW_PAIRED__Z2 <= 0
 c4159: aux_cap_helper__b2,r11,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b2,r11,band_HIGH_PAIRED__condX2 <= 0
 c4160: - aux_cap_helper__b9,r4,band_UNPAIRED__Z1
        + aux_cap_helper__b9,r4,band_UNPAIRED__condX1 <= 0
 c4161: - aux_Omega_b3,r11 + 835.640474113585 aux_sv_b3,r11  = 0
 c4162: sv_function_helpervar_b0,r4__Z1 + sv_function_helpervar_b0,r4__Z2
        + sv_function_helpervar_b0,r4__Z3  = 1
 c4163: GlobalBidder_W_i_7 + 14 GlobalBidder_hat{W}__i7,k1 <= 27
 c4164: - sv_function_helpervar_b1,r1__condX1 <= 0
 c4165: 4.4504289328 aux_cap_helper__b9,r3,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b9,r3,band_LOW_PAIRED__condX1
        - 1.2715511236 aux_cap_helper__b9,r3,band_LOW_PAIRED__Z2
        - aux_cap_b9,r3,band_LOW_PAIRED  = 0
 c4166: - 3 aux_cap_helper__b4,r5,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b4,r5,band_HIGH_PAIRED__condX3 <= 0
 c4167: - sv_function_helpervar_b8,r11__condX3
        + 7.69516567267345 sv_function_helpervar_b8,r11__Z3 <= 0
 c4168: - sv_function_helpervar_b3,r9__condX1 <= 0
 c4169: - aux_c_b6,r12 + sv_function_helpervar_b6,r12__condX1
        + sv_function_helpervar_b6,r12__condX2
        + sv_function_helpervar_b6,r12__condX3  = 0
 c4170: aux_Omega_b9,r1 + aux_Omega_b9,r0 + aux_Omega_b9,r3 + aux_Omega_b9,r2
        + aux_Omega_b9,r5 + aux_Omega_b9,r4 + aux_Omega_b9,r7 + aux_Omega_b9,r6
        + aux_Omega_b9,r9 + aux_Omega_b9,r8 - GlobalBidder_PSI__i9,k1
        + aux_Omega_b9,r13 + aux_Omega_b9,r10 + aux_Omega_b9,r11
        + aux_Omega_b9,r12 + 20668026.8748422 GlobalBidder_hat{W}__i9,k1
        <= 20668026.8748422
 c4171: sv_function_helpervar_b3,r10__condX1
        - 0.158453086896816 sv_function_helpervar_b3,r10__Z1 <= 0
 c4172: aux_cap_helper__b9,r0,band_UNPAIRED__condX1
        + aux_cap_helper__b9,r0,band_UNPAIRED__condX2 - X__b9,r0,band_UNPAIRED
         = 0
 c4173: sv_function_helpervar_b2,r4__condX2
        - 6.2359236486132 sv_function_helpervar_b2,r4__Z2 <= 0
 c4174: sv_function_helpervar_b7,r11__condX2
        - 7.94617793150088 sv_function_helpervar_b7,r11__Z2 <= 0
 c4175: - 1.2715511236 aux_cap_helper__b3,r2,band_LOW_PAIRED__Z2
        + 4.4504289328 aux_cap_helper__b3,r2,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b3,r2,band_LOW_PAIRED__condX1
        - aux_cap_b3,r2,band_LOW_PAIRED  = 0
 c4176: 1.21690011250031 aux_cap_helper__b5,r10,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b5,r10,band_HIGH_PAIRED__Z2
        - aux_cap_b5,r10,band_HIGH_PAIRED
        + 2.0281668543 aux_cap_helper__b5,r10,band_HIGH_PAIRED__condX1
        + 1.8253501688 aux_cap_helper__b5,r10,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b5,r10,band_HIGH_PAIRED__condX2  = 0
 c4177: - 2 aux_cap_helper__b6,r12,band_UNPAIRED__Z2
        + aux_cap_helper__b6,r12,band_UNPAIRED__condX2 <= 0
 c4178: - aux_cap_helper__b9,r1,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b9,r1,band_HIGH_PAIRED__Z3 <= 0
 c4179: - 5.52659420777561 sv_function_helpervar_b4,r3__Z2
        + sv_function_helpervar_b4,r3__condX2 <= 0
 c4180: aux_cap_helper__b8,r8,band_LOW_PAIRED__condX1
        + aux_cap_helper__b8,r8,band_LOW_PAIRED__condX2
        - X__b8,r8,band_LOW_PAIRED  = 0
 c4181: aux_cap_helper__b7,r3,band_LOW_PAIRED__Z1
        + aux_cap_helper__b7,r3,band_LOW_PAIRED__Z2  = 1
 c4182: - aux_cap_helper__b5,r9,band_LOW_PAIRED__Z1
        + aux_cap_helper__b5,r9,band_LOW_PAIRED__condX1 <= 0
 c4183: - aux_cap_helper__b9,r11,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b9,r11,band_HIGH_PAIRED__Z3 <= 0
 c4184: - sv_function_helpervar_b4,r8__condX2
        + 0.158453086912977 sv_function_helpervar_b4,r8__Z2 <= 0
 c4185: aux_cap_helper__b2,r6,band_UNPAIRED__Z1
        + aux_cap_helper__b2,r6,band_UNPAIRED__Z2  = 1
 c4186: 399.861973954658 aux_sv_b1,r10 - aux_Omega_b1,r10  = 0
 c4187: aux_cap_helper__b2,r8,band_UNPAIRED__Z1
        + aux_cap_helper__b2,r8,band_UNPAIRED__Z2  = 1
 c4188: - aux_cap_helper__b3,r5,band_LOW_PAIRED__Z1
        + aux_cap_helper__b3,r5,band_LOW_PAIRED__condX1 <= 0
 c4189: - 2 aux_cap_helper__b6,r1,band_LOW_PAIRED__Z2
        + aux_cap_helper__b6,r1,band_LOW_PAIRED__condX2 <= 0
 c4190: - sv_function_helpervar_b2,r4__condX1 <= 0
 c4191: - aux_cap_helper__b1,r1,band_LOW_PAIRED__Z1
        + aux_cap_helper__b1,r1,band_LOW_PAIRED__condX1 <= 0
 c4192: - 15.8453086887446 sv_function_helpervar_b0,r11__Z3
        + sv_function_helpervar_b0,r11__condX3 <= 0
 c4193: - 2 aux_cap_helper__b4,r7,band_UNPAIRED__Z2
        + aux_cap_helper__b4,r7,band_UNPAIRED__condX2 <= 0
 c4194: aux_cap_helper__b6,r8,band_LOW_PAIRED__condX1
        + aux_cap_helper__b6,r8,band_LOW_PAIRED__condX2
        - X__b6,r8,band_LOW_PAIRED  = 0
 c4195: 0.15845308693176 sv_function_helpervar_b6,r0__Z2
        - sv_function_helpervar_b6,r0__condX2 <= 0
 c4196: - sv_function_helpervar_b7,r6__condX1 <= 0
 c4197: aux_cap_b4,r5,band_UNPAIRED + aux_cap_b4,r5,band_LOW_PAIRED
        + aux_cap_b4,r5,band_HIGH_PAIRED - aux_c_b4,r5  = 0
 c4198: - aux_cap_helper__b6,r1,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b6,r1,band_HIGH_PAIRED__condX1 <= 0
 c4199: - sv_function_helpervar_b6,r12__condX1 <= 0
 c4200: 2 aux_cap_helper__b4,r2,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b4,r2,band_HIGH_PAIRED__condX3 <= 0
 c4201: - 15.8453086887446 sv_function_helpervar_b6,r4__Z3
        + sv_function_helpervar_b6,r4__condX3 <= 0
 c4202: - 2 aux_cap_helper__b0,r13,band_UNPAIRED__Z2
        + aux_cap_helper__b0,r13,band_UNPAIRED__condX2 <= 0
 c4203: - 3 aux_cap_helper__b7,r5,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b7,r5,band_HIGH_PAIRED__condX3 <= 0
 c4204: aux_cap_helper__b7,r8,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b7,r8,band_HIGH_PAIRED__Z2 <= 0
 c4205: sv_function_helpervar_b1,r2__condX3
        - 15.8453086887446 sv_function_helpervar_b1,r2__Z3 <= 0
 c4206: - sv_function_helpervar_b3,r8__condX3
        + 6.12592415881159 sv_function_helpervar_b3,r8__Z3 <= 0
 c4207: 0.888446608 aux_cap_helper__b5,r13,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b5,r13,band_UNPAIRED__condX1
        - 0.253841888 aux_cap_helper__b5,r13,band_UNPAIRED__Z2
        - aux_cap_b5,r13,band_UNPAIRED  = 0
 c4208: aux_cap_helper__b6,r6,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b6,r6,band_UNPAIRED__Z2 <= 0
 c4209: - aux_cap_helper__b8,r0,band_LOW_PAIRED__condX1 <= 0
 c4210: - aux_cap_helper__b9,r8,band_UNPAIRED__condX2
        + aux_cap_helper__b9,r8,band_UNPAIRED__Z2 <= 0
 c4211: X__b7,r13,band_UNPAIRED + X__b8,r13,band_UNPAIRED
        + X__b4,r13,band_UNPAIRED + X__b2,r13,band_UNPAIRED
        + X__b0,r13,band_UNPAIRED + X__b6,r13,band_UNPAIRED
        + X__b3,r13,band_UNPAIRED + X__b9,r13,band_UNPAIRED
        + X__b1,r13,band_UNPAIRED + X__b5,r13,band_UNPAIRED <= 2
 c4212: sv_function_helpervar_b6,r13__condX3
        - 15.8453086887446 sv_function_helpervar_b6,r13__Z3 <= 0
 c4213: X__b0,r2,band_HIGH_PAIRED + X__b7,r2,band_HIGH_PAIRED
        + X__b4,r2,band_HIGH_PAIRED + X__b1,r2,band_HIGH_PAIRED
        + X__b3,r2,band_HIGH_PAIRED + X__b9,r2,band_HIGH_PAIRED
        + X__b2,r2,band_HIGH_PAIRED + X__b5,r2,band_HIGH_PAIRED
        + X__b6,r2,band_HIGH_PAIRED + X__b8,r2,band_HIGH_PAIRED <= 3
 c4214: aux_cap_helper__b8,r6,band_LOW_PAIRED__condX2
        + aux_cap_helper__b8,r6,band_LOW_PAIRED__condX1
        - X__b8,r6,band_LOW_PAIRED  = 0
 c4215: 22.8966411555837 sv_function_helpervar_b1,r0__Z2
        + 48.1199812155538 sv_function_helpervar_b1,r0__Z3 - aux_sv_b1,r0
        + 6.446532658 sv_function_helpervar_b1,r0__condX2
        + 2.5537922651 sv_function_helpervar_b1,r0__condX3
        + 150.9476061544 sv_function_helpervar_b1,r0__condX1  = 0
 c4216: 515.767533017734 sv_function_helpervar_b8,r5__Z3
        + 248.425409287123 sv_function_helpervar_b8,r5__Z2 - aux_sv_b8,r5
        + 1636.3575441861 sv_function_helpervar_b8,r5__condX1
        + 68.5407590325 sv_function_helpervar_b8,r5__condX2
        + 28.0556623524 sv_function_helpervar_b8,r5__condX3  = 0
 c4217: - 0.253841888 aux_cap_helper__b8,r0,band_UNPAIRED__Z2
        + 0.63460472 aux_cap_helper__b8,r0,band_UNPAIRED__condX1
        + 0.888446608 aux_cap_helper__b8,r0,band_UNPAIRED__condX2
        - aux_cap_b8,r0,band_UNPAIRED  = 0
 c4218: aux_cap_helper__b4,r9,band_UNPAIRED__condX1
        + aux_cap_helper__b4,r9,band_UNPAIRED__condX2 - X__b4,r9,band_UNPAIRED
         = 0
 c4219: - 2 aux_cap_helper__b5,r5,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b5,r5,band_HIGH_PAIRED__condX2 <= 0
 c4220: aux_cap_helper__b0,r11,band_LOW_PAIRED__condX2
        + aux_cap_helper__b0,r11,band_LOW_PAIRED__condX1
        - X__b0,r11,band_LOW_PAIRED  = 0
 c4221: - sv_function_helpervar_b8,r9__condX1 <= 0
 c4222: - sv_function_helpervar_b2,r6__condX1 <= 0
 c4223: aux_cap_helper__b5,r10,band_UNPAIRED__Z2
        - aux_cap_helper__b5,r10,band_UNPAIRED__condX2 <= 0
 c4224: - 15.8453086887446 sv_function_helpervar_b2,r7__Z3
        + sv_function_helpervar_b2,r7__condX3 <= 0
 c4225: - aux_cap_b4,r4,band_LOW_PAIRED
        + 3.1788778092 aux_cap_helper__b4,r4,band_LOW_PAIRED__condX1
        + 4.4504289328 aux_cap_helper__b4,r4,band_LOW_PAIRED__condX2
        - 1.2715511236 aux_cap_helper__b4,r4,band_LOW_PAIRED__Z2  = 0
 c4226: - 0.253841888 aux_cap_helper__b0,r1,band_UNPAIRED__Z2
        - aux_cap_b0,r1,band_UNPAIRED
        + 0.63460472 aux_cap_helper__b0,r1,band_UNPAIRED__condX1
        + 0.888446608 aux_cap_helper__b0,r1,band_UNPAIRED__condX2  = 0
 c4227: 0.15845308698947 sv_function_helpervar_b1,r1__Z2
        - sv_function_helpervar_b1,r1__condX2 <= 0
 c4228: aux_cap_helper__b2,r5,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b2,r5,band_HIGH_PAIRED__Z2 <= 0
 c4229: - sv_function_helpervar_b7,r3__condX3
        + 6.99638282066709 sv_function_helpervar_b7,r3__Z3 <= 0
 c4230: sv_function_helpervar_b8,r8__condX1
        - 0.158453086894935 sv_function_helpervar_b8,r8__Z1 <= 0
 c4231: aux_cap_helper__b1,r9,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b1,r9,band_LOW_PAIRED__Z2 <= 0
 c4232: aux_cap_helper__b1,r13,band_LOW_PAIRED__condX1
        + aux_cap_helper__b1,r13,band_LOW_PAIRED__condX2
        - X__b1,r13,band_LOW_PAIRED  = 0
 c4233: - aux_cap_helper__b5,r10,band_UNPAIRED__condX1 <= 0
 c4234: - aux_cap_helper__b4,r8,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b4,r8,band_HIGH_PAIRED__Z3 <= 0
 c4235: 0.158453086896477 sv_function_helpervar_b8,r9__Z2
        - sv_function_helpervar_b8,r9__condX2 <= 0
 c4236: sv_function_helpervar_b1,r4__condX1
        + sv_function_helpervar_b1,r4__condX2
        + sv_function_helpervar_b1,r4__condX3 - aux_c_b1,r4  = 0
 c4237: - X__b3,r6,band_UNPAIRED + aux_cap_helper__b3,r6,band_UNPAIRED__condX1
        + aux_cap_helper__b3,r6,band_UNPAIRED__condX2  = 0
 c4238: - aux_cap_helper__b8,r0,band_LOW_PAIRED__condX2
        + aux_cap_helper__b8,r0,band_LOW_PAIRED__Z2 <= 0
 c4239: - aux_Omega_b2,r2 + 142.447650872583 aux_sv_b2,r2  = 0
 c4240: - aux_cap_helper__b5,r11,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b5,r11,band_HIGH_PAIRED__condX1 <= 0
 c4241: aux_cap_helper__b3,r10,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b3,r10,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b3,r10,band_HIGH_PAIRED__condX3
        - X__b3,r10,band_HIGH_PAIRED  = 0
 c4242: - aux_Omega_b7,r1 + 364.655927605161 aux_sv_b7,r1  = 0
 c4243: aux_cap_helper__b0,r11,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b0,r11,band_HIGH_PAIRED__Z3 <= 0
 c4244: aux_cap_helper__b1,r5,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b1,r5,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b1,r5,band_HIGH_PAIRED__Z2  = 1
 c4245: - 2 aux_cap_helper__b0,r4,band_LOW_PAIRED__Z2
        + aux_cap_helper__b0,r4,band_LOW_PAIRED__condX2 <= 0
 c4246: - aux_cap_helper__b1,r10,band_HIGH_PAIRED__condX1 <= 0
 c4247: aux_cap_helper__b3,r1,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b3,r1,band_HIGH_PAIRED__Z1 <= 0
 c4248: - aux_cap_b5,r11,band_UNPAIRED
        + 0.63460472 aux_cap_helper__b5,r11,band_UNPAIRED__condX1
        + 0.888446608 aux_cap_helper__b5,r11,band_UNPAIRED__condX2
        - 0.253841888 aux_cap_helper__b5,r11,band_UNPAIRED__Z2  = 0
 c4249: - aux_cap_helper__b7,r2,band_UNPAIRED__condX2
        + aux_cap_helper__b7,r2,band_UNPAIRED__Z2 <= 0
 c4250: sv_function_helpervar_b5,r13__Z3 + sv_function_helpervar_b5,r13__Z2
        + sv_function_helpervar_b5,r13__Z1  = 1
 c4251: aux_cap_helper__b6,r4,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b6,r4,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b6,r4,band_HIGH_PAIRED__Z1  = 1
 c4252: sv_function_helpervar_b1,r6__condX1
        + sv_function_helpervar_b1,r6__condX3
        + sv_function_helpervar_b1,r6__condX2 - aux_c_b1,r6  = 0
 c4253: - aux_cap_helper__b3,r13,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b3,r13,band_HIGH_PAIRED__Z2 <= 0
 c4254: aux_cap_b1,r4,band_LOW_PAIRED + aux_cap_b1,r4,band_UNPAIRED
        + aux_cap_b1,r4,band_HIGH_PAIRED - aux_c_b1,r4  = 0
 c4255: - X__b0,r7,band_UNPAIRED + aux_cap_helper__b0,r7,band_UNPAIRED__condX2
        + aux_cap_helper__b0,r7,band_UNPAIRED__condX1  = 0
 c4256: - 2 aux_cap_helper__b8,r5,band_UNPAIRED__Z2
        + aux_cap_helper__b8,r5,band_UNPAIRED__condX2 <= 0
 c4257: - 6.26935260585317 sv_function_helpervar_b2,r11__Z2
        + sv_function_helpervar_b2,r11__condX2 <= 0
 c4258: - aux_cap_helper__b9,r8,band_UNPAIRED__condX1 <= 0
 c4259: aux_cap_helper__b6,r11,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b6,r11,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b6,r11,band_HIGH_PAIRED__Z1  = 1
 c4260: 4.4504289328 aux_cap_helper__b4,r5,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b4,r5,band_LOW_PAIRED__condX1
        - aux_cap_b4,r5,band_LOW_PAIRED
        - 1.2715511236 aux_cap_helper__b4,r5,band_LOW_PAIRED__Z2  = 0
 c4261: - aux_cap_helper__b1,r3,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b1,r3,band_HIGH_PAIRED__Z2 <= 0
 c4262: sv_function_helpervar_b3,r3__condX3
        + sv_function_helpervar_b3,r3__condX2
        + sv_function_helpervar_b3,r3__condX1 - aux_c_b3,r3  = 0
 c4263: - X__b9,r8,band_HIGH_PAIRED
        + aux_cap_helper__b9,r8,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b9,r8,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b9,r8,band_HIGH_PAIRED__condX3  = 0
 c4264: - aux_Omega_b2,r3 + 466.912217158911 aux_sv_b2,r3  = 0
 c4265: - aux_cap_helper__b6,r6,band_LOW_PAIRED__Z1
        + aux_cap_helper__b6,r6,band_LOW_PAIRED__condX1 <= 0
 c4266: aux_cap_helper__b4,r13,band_LOW_PAIRED__Z2
        + aux_cap_helper__b4,r13,band_LOW_PAIRED__Z1  = 1
 c4267: - 2 aux_cap_helper__b9,r2,band_LOW_PAIRED__Z2
        + aux_cap_helper__b9,r2,band_LOW_PAIRED__condX2 <= 0
 c4268: - aux_cap_helper__b4,r2,band_LOW_PAIRED__Z1
        + aux_cap_helper__b4,r2,band_LOW_PAIRED__condX1 <= 0
 c4269: - 5.74546828281525 sv_function_helpervar_b5,r3__Z2
        + sv_function_helpervar_b5,r3__condX2 <= 0
 c4270: - aux_cap_helper__b1,r3,band_HIGH_PAIRED__condX1 <= 0
 c4271: aux_cap_b0,r6,band_HIGH_PAIRED - aux_c_b0,r6
        + aux_cap_b0,r6,band_LOW_PAIRED + aux_cap_b0,r6,band_UNPAIRED  = 0
 c4272: aux_cap_b9,r5,band_HIGH_PAIRED + aux_cap_b9,r5,band_UNPAIRED
        + aux_cap_b9,r5,band_LOW_PAIRED - aux_c_b9,r5  = 0
 c4273: sv_function_helpervar_b7,r5__Z1 + sv_function_helpervar_b7,r5__Z2
        + sv_function_helpervar_b7,r5__Z3  = 1
 c4274: aux_cap_helper__b1,r10,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b1,r10,band_HIGH_PAIRED__condX2 <= 0
 c4275: aux_cap_helper__b6,r12,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b6,r12,band_HIGH_PAIRED__Z1 <= 0
 c4276: - 0.158453086894239 sv_function_helpervar_b5,r13__Z1
        + sv_function_helpervar_b5,r13__condX1 <= 0
 c4277: - 0.15845308691419 sv_function_helpervar_b4,r0__Z1
        + sv_function_helpervar_b4,r0__condX1 <= 0
 c4278: - aux_cap_helper__b7,r9,band_UNPAIRED__condX1 <= 0
 c4279: - aux_sv_b8,r8 + 27.6586256188 sv_function_helpervar_b8,r8__condX3
        + 69.9812688481 sv_function_helpervar_b8,r8__condX2
        + 1636.3575441455 sv_function_helpervar_b8,r8__condX1
        + 522.058702622724 sv_function_helpervar_b8,r8__Z3
        + 248.197156059864 sv_function_helpervar_b8,r8__Z2  = 0
 c4280: aux_cap_helper__b9,r13,band_UNPAIRED__condX2
        + aux_cap_helper__b9,r13,band_UNPAIRED__condX1
        - X__b9,r13,band_UNPAIRED  = 0
 c4281: - sv_function_helpervar_b9,r3__condX2
        + 0.158453086900186 sv_function_helpervar_b9,r3__Z2 <= 0
 c4282: - aux_Omega_b3,r5 + 289.503148041286 aux_sv_b3,r5  = 0
 c4283: - sv_function_helpervar_b5,r8__condX2
        + 0.158453086900078 sv_function_helpervar_b5,r8__Z2 <= 0
 c4284: X__b7,r1,band_HIGH_PAIRED + X__b7,r1,band_LOW_PAIRED
        - GlobalBidder_W_ir__b7,r1 + X__b7,r1,band_UNPAIRED >= 0
 c4285: aux_cap_b3,r7,band_UNPAIRED + aux_cap_b3,r7,band_LOW_PAIRED
        + aux_cap_b3,r7,band_HIGH_PAIRED - aux_c_b3,r7  = 0
 c4286: - 2 aux_cap_helper__b9,r1,band_UNPAIRED__Z2
        + aux_cap_helper__b9,r1,band_UNPAIRED__condX2 <= 0
 c4287: aux_cap_helper__b6,r8,band_UNPAIRED__condX1
        + aux_cap_helper__b6,r8,band_UNPAIRED__condX2 - X__b6,r8,band_UNPAIRED
         = 0
 c4288: sv_function_helpervar_b7,r6__Z2 + sv_function_helpervar_b7,r6__Z3
        + sv_function_helpervar_b7,r6__Z1  = 1
 c4289: 3792.68997605925 aux_sv_b8,r7 - aux_Omega_b8,r7  = 0
 c4290: sv_function_helpervar_b7,r7__condX3
        - 15.8453086887446 sv_function_helpervar_b7,r7__Z3 <= 0
 c4291: 0.158453086899434 sv_function_helpervar_b3,r6__Z2
        - sv_function_helpervar_b3,r6__condX2 <= 0
 c4292: - aux_cap_helper__b0,r1,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b0,r1,band_HIGH_PAIRED__condX1 <= 0
 c4293: 0.158453086898492 sv_function_helpervar_b0,r11__Z2
        - sv_function_helpervar_b0,r11__condX2 <= 0
 c4294: sv_function_helpervar_b3,r13__condX2
        - 5.48961824768186 sv_function_helpervar_b3,r13__Z2 <= 0
 c4295: aux_cap_helper__b1,r13,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b1,r13,band_HIGH_PAIRED__Z3 <= 0
 c4296: - sv_function_helpervar_b9,r5__condX3
        + 7.77426771339398 sv_function_helpervar_b9,r5__Z3 <= 0
 c4297: aux_cap_helper__b1,r2,band_LOW_PAIRED__condX2
        + aux_cap_helper__b1,r2,band_LOW_PAIRED__condX1
        - X__b1,r2,band_LOW_PAIRED  = 0
 c4298: - aux_cap_helper__b2,r3,band_LOW_PAIRED__condX1 <= 0
 c4299: aux_cap_helper__b7,r8,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b7,r8,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b7,r8,band_HIGH_PAIRED__condX3
        - X__b7,r8,band_HIGH_PAIRED  = 0
 c4300: X__b7,r11,band_UNPAIRED + X__b7,r11,band_LOW_PAIRED
        + X__b7,r11,band_HIGH_PAIRED - GlobalBidder_W_ir__b7,r11 >= 0
 c4301: 150.9476061485 sv_function_helpervar_b1,r4__condX1
        + 7.0525279188 sv_function_helpervar_b1,r4__condX2
        + 2.413807585 sv_function_helpervar_b1,r4__condX3
        + 50.338081682854 sv_function_helpervar_b1,r4__Z3
        + 22.8006193358279 sv_function_helpervar_b1,r4__Z2 - aux_sv_b1,r4  = 0
 c4302: 5.94902634456456 sv_function_helpervar_b3,r12__Z3
        - sv_function_helpervar_b3,r12__condX3 <= 0
 c4303: - aux_Omega_b3,r1 + 299.782355390873 aux_sv_b3,r1  = 0
 c4304: aux_cap_helper__b9,r7,band_UNPAIRED__condX2
        + aux_cap_helper__b9,r7,band_UNPAIRED__condX1 - X__b9,r7,band_UNPAIRED
         = 0
 c4305: aux_cap_helper__b5,r2,band_UNPAIRED__Z2
        + aux_cap_helper__b5,r2,band_UNPAIRED__Z1  = 1
 c4306: aux_cap_helper__b7,r3,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b7,r3,band_LOW_PAIRED__Z2 <= 0
 c4307: X__b9,r4,band_UNPAIRED + X__b9,r4,band_HIGH_PAIRED
        - GlobalBidder_W_ir__b9,r4 + X__b9,r4,band_LOW_PAIRED >= 0
 c4308: - aux_cap_helper__b7,r9,band_UNPAIRED__condX2
        + aux_cap_helper__b7,r9,band_UNPAIRED__Z2 <= 0
 c4309: - 2 aux_cap_helper__b8,r10,band_UNPAIRED__Z2
        + aux_cap_helper__b8,r10,band_UNPAIRED__condX2 <= 0
 c4310: aux_cap_helper__b8,r4,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b8,r4,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b8,r4,band_HIGH_PAIRED__Z1  = 1
 c4311: aux_cap_helper__b3,r12,band_LOW_PAIRED__condX1
        + aux_cap_helper__b3,r12,band_LOW_PAIRED__condX2
        - X__b3,r12,band_LOW_PAIRED  = 0
 c4312: 36.0365617243137 sv_function_helpervar_b2,r12__Z3
        + 17.4200488558028 sv_function_helpervar_b2,r12__Z2 - aux_sv_b2,r12
        + 4.778483228 sv_function_helpervar_b2,r12__condX2
        + 1.9744931904 sv_function_helpervar_b2,r12__condX3
        + 114.7166939435 sv_function_helpervar_b2,r12__condX1  = 0
 c4313: 0.158453086927404 sv_function_helpervar_b0,r8__Z2
        - sv_function_helpervar_b0,r8__condX2 <= 0
 c4314: aux_cap_helper__b7,r10,band_UNPAIRED__Z2
        + aux_cap_helper__b7,r10,band_UNPAIRED__Z1  = 1
 c4315: aux_cap_helper__b8,r1,band_LOW_PAIRED__Z2
        + aux_cap_helper__b8,r1,band_LOW_PAIRED__Z1  = 1
 c4316: - 2 aux_cap_helper__b6,r3,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b6,r3,band_HIGH_PAIRED__condX2 <= 0
 c4317: aux_cap_helper__b1,r11,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b1,r11,band_HIGH_PAIRED__Z2 <= 0
 c4318: - 2 aux_cap_helper__b8,r9,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b8,r9,band_HIGH_PAIRED__condX2 <= 0
 c4319: X__b9,r9,band_LOW_PAIRED + X__b9,r9,band_UNPAIRED
        + X__b9,r9,band_HIGH_PAIRED - GlobalBidder_W_ir__b9,r9 >= 0
 c4320: 4.4504289328 aux_cap_helper__b5,r3,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b5,r3,band_LOW_PAIRED__condX1
        - aux_cap_b5,r3,band_LOW_PAIRED
        - 1.2715511236 aux_cap_helper__b5,r3,band_LOW_PAIRED__Z2  = 0
 c4321: - 15.8453086887446 sv_function_helpervar_b5,r7__Z3
        + sv_function_helpervar_b5,r7__condX3 <= 0
 c4322: aux_Omega_b9,r1 + aux_Omega_b9,r0 + aux_Omega_b9,r3 + aux_Omega_b9,r2
        + aux_Omega_b9,r5 + aux_Omega_b9,r4 + aux_Omega_b9,r7 + aux_Omega_b9,r6
        + aux_Omega_b9,r9 + aux_Omega_b9,r8 - GlobalBidder_PSI__i9,k2
        + aux_Omega_b9,r13 + aux_Omega_b9,r10 + aux_Omega_b9,r11
        + aux_Omega_b9,r12 - 20668026.8748422 GlobalBidder_hat{W}__i9,k2
        >= -20668026.8748422
 c4323: - 7.60706438507882 sv_function_helpervar_b8,r9__Z2
        + sv_function_helpervar_b8,r9__condX2 <= 0
 c4324: - aux_cap_helper__b9,r8,band_LOW_PAIRED__condX1 <= 0
 c4325: sv_function_helpervar_b4,r10__condX1
        - 0.158453086950738 sv_function_helpervar_b4,r10__Z1 <= 0
 c4326: aux_cap_helper__b6,r13,band_UNPAIRED__Z2
        - aux_cap_helper__b6,r13,band_UNPAIRED__condX2 <= 0
 c4327: aux_cap_helper__b1,r11,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b1,r11,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b1,r11,band_HIGH_PAIRED__Z2  = 1
 c4328: aux_cap_helper__b5,r11,band_UNPAIRED__condX1
        + aux_cap_helper__b5,r11,band_UNPAIRED__condX2
        - X__b5,r11,band_UNPAIRED  = 0
 c4329: - sv_function_helpervar_b7,r5__condX1 <= 0
 c4330: aux_cap_helper__b0,r4,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b0,r4,band_HIGH_PAIRED__Z3 <= 0
 c4331: - aux_c_b2,r0 + sv_function_helpervar_b2,r0__condX1
        + sv_function_helpervar_b2,r0__condX2
        + sv_function_helpervar_b2,r0__condX3  = 0
 c4332: aux_cap_b2,r13,band_UNPAIRED + aux_cap_b2,r13,band_HIGH_PAIRED
        - aux_c_b2,r13 + aux_cap_b2,r13,band_LOW_PAIRED  = 0
 c4333: 2 aux_cap_helper__b4,r12,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b4,r12,band_HIGH_PAIRED__condX3 <= 0
 c4334: - aux_cap_helper__b8,r10,band_LOW_PAIRED__condX1 <= 0
 c4335: aux_cap_helper__b1,r11,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b1,r11,band_UNPAIRED__Z2 <= 0
 c4336: - aux_cap_helper__b6,r13,band_UNPAIRED__condX1 <= 0
 c4337: sv_function_helpervar_b2,r12__Z3 + sv_function_helpervar_b2,r12__Z2
        + sv_function_helpervar_b2,r12__Z1  = 1
 c4338: - aux_cap_helper__b0,r13,band_LOW_PAIRED__Z1
        + aux_cap_helper__b0,r13,band_LOW_PAIRED__condX1 <= 0
 c4339: - aux_cap_b3,r12,band_HIGH_PAIRED
        - 0.8112667416 aux_cap_helper__b3,r12,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b3,r12,band_HIGH_PAIRED__Z3
        + 1.8253501688 aux_cap_helper__b3,r12,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b3,r12,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b3,r12,band_HIGH_PAIRED__condX1  = 0
 c4340: sv_function_helpervar_b5,r1__condX1
        - 0.158453086936865 sv_function_helpervar_b5,r1__Z1 <= 0
 c4341: aux_cap_helper__b6,r12,band_UNPAIRED__Z2
        + aux_cap_helper__b6,r12,band_UNPAIRED__Z1  = 1
 c4342: 0.888446608 aux_cap_helper__b9,r5,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b9,r5,band_UNPAIRED__condX1
        - 0.253841888 aux_cap_helper__b9,r5,band_UNPAIRED__Z2
        - aux_cap_b9,r5,band_UNPAIRED  = 0
 c4343: - 0.253841888 aux_cap_helper__b3,r1,band_UNPAIRED__Z2
        + 0.888446608 aux_cap_helper__b3,r1,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b3,r1,band_UNPAIRED__condX1
        - aux_cap_b3,r1,band_UNPAIRED  = 0
 c4344: - 0.253841888 aux_cap_helper__b6,r3,band_UNPAIRED__Z2
        + 0.888446608 aux_cap_helper__b6,r3,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b6,r3,band_UNPAIRED__condX1
        - aux_cap_b6,r3,band_UNPAIRED  = 0
 c4345: - aux_cap_helper__b3,r7,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b3,r7,band_HIGH_PAIRED__Z3 <= 0
 c4346: 17.6296988209 sv_function_helpervar_b3,r13__condX3
        + 58.3439573431 sv_function_helpervar_b3,r13__condX2
        + 1152.1877364187 sv_function_helpervar_b3,r13__condX1 - aux_sv_b3,r13
        + 173.322923413003 sv_function_helpervar_b3,r13__Z2
        + 396.828659937303 sv_function_helpervar_b3,r13__Z3  = 0
 c4347: 215.198549875405 aux_sv_b0,r8 - aux_Omega_b0,r8  = 0
 c4348: - X__b5,r7,band_HIGH_PAIRED
        + aux_cap_helper__b5,r7,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b5,r7,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b5,r7,band_HIGH_PAIRED__condX1  = 0
 c4349: sv_function_helpervar_b1,r9__Z3 + sv_function_helpervar_b1,r9__Z2
        + sv_function_helpervar_b1,r9__Z1  = 1
 c4350: - aux_cap_helper__b4,r10,band_UNPAIRED__condX1 <= 0
 c4351: - aux_cap_b7,r7,band_LOW_PAIRED
        - 1.2715511236 aux_cap_helper__b7,r7,band_LOW_PAIRED__Z2
        + 4.4504289328 aux_cap_helper__b7,r7,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b7,r7,band_LOW_PAIRED__condX1  = 0
 c4352: aux_cap_helper__b9,r8,band_LOW_PAIRED__Z2
        - aux_cap_helper__b9,r8,band_LOW_PAIRED__condX2 <= 0
 c4353: aux_cap_helper__b3,r7,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b3,r7,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b3,r7,band_HIGH_PAIRED__Z2  = 1
 c4354: - 2 aux_cap_helper__b1,r1,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b1,r1,band_HIGH_PAIRED__condX2 <= 0
 c4355: aux_cap_helper__b7,r4,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b7,r4,band_UNPAIRED__Z2 <= 0
 c4356: aux_cap_b2,r8,band_LOW_PAIRED + aux_cap_b2,r8,band_HIGH_PAIRED
        - aux_c_b2,r8 + aux_cap_b2,r8,band_UNPAIRED  = 0
 c4357: aux_cap_helper__b8,r10,band_LOW_PAIRED__Z2
        - aux_cap_helper__b8,r10,band_LOW_PAIRED__condX2 <= 0
 c4358: sv_function_helpervar_b4,r2__condX1
        - 0.158453086926367 sv_function_helpervar_b4,r2__Z1 <= 0
 c4359: aux_cap_helper__b8,r10,band_LOW_PAIRED__Z1
        + aux_cap_helper__b8,r10,band_LOW_PAIRED__Z2  = 1
 c4360: - X__b8,r3,band_HIGH_PAIRED
        + aux_cap_helper__b8,r3,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b8,r3,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b8,r3,band_HIGH_PAIRED__condX1  = 0
 c4361: aux_cap_helper__b2,r10,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b2,r10,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b2,r10,band_HIGH_PAIRED__Z3  = 1
 c4362: sv_function_helpervar_b8,r2__Z3 + sv_function_helpervar_b8,r2__Z2
        + sv_function_helpervar_b8,r2__Z1  = 1
 c4363: - 2 aux_cap_helper__b6,r4,band_LOW_PAIRED__Z2
        + aux_cap_helper__b6,r4,band_LOW_PAIRED__condX2 <= 0
 c4364: aux_cap_b8,r10,band_HIGH_PAIRED - aux_c_b8,r10
        + aux_cap_b8,r10,band_UNPAIRED + aux_cap_b8,r10,band_LOW_PAIRED  = 0
 c4365: - 15.8453086887446 sv_function_helpervar_b9,r2__Z3
        + sv_function_helpervar_b9,r2__condX3 <= 0
 c4366: aux_cap_b6,r0,band_UNPAIRED + aux_cap_b6,r0,band_HIGH_PAIRED
        - aux_c_b6,r0 + aux_cap_b6,r0,band_LOW_PAIRED  = 0
 c4367: aux_cap_helper__b4,r2,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b4,r2,band_UNPAIRED__Z2 <= 0
 c4368: - X__b7,r4,band_LOW_PAIRED
        + aux_cap_helper__b7,r4,band_LOW_PAIRED__condX2
        + aux_cap_helper__b7,r4,band_LOW_PAIRED__condX1  = 0
 c4369: aux_cap_helper__b4,r9,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b4,r9,band_HIGH_PAIRED__Z1 <= 0
 c4370: - X__b8,r3,band_UNPAIRED + aux_cap_helper__b8,r3,band_UNPAIRED__condX2
        + aux_cap_helper__b8,r3,band_UNPAIRED__condX1  = 0
 c4371: aux_cap_helper__b3,r1,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b3,r1,band_HIGH_PAIRED__Z3 <= 0
 c4372: - aux_Omega_b5,r1 + 149.392208717655 aux_sv_b5,r1  = 0
 c4373: - 6.15627515730785 sv_function_helpervar_b5,r13__Z2
        + sv_function_helpervar_b5,r13__condX2 <= 0
 c4374: - 3 aux_cap_helper__b5,r7,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b5,r7,band_HIGH_PAIRED__condX3 <= 0
 c4375: sv_function_helpervar_b4,r9__condX3
        - 15.8453086887446 sv_function_helpervar_b4,r9__Z3 <= 0
 c4376: - aux_cap_helper__b5,r0,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b5,r0,band_HIGH_PAIRED__condX1 <= 0
 c4377: aux_cap_helper__b4,r0,band_UNPAIRED__Z2
        + aux_cap_helper__b4,r0,band_UNPAIRED__Z1  = 1
 c4378: aux_cap_helper__b7,r6,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b7,r6,band_HIGH_PAIRED__Z1 <= 0
 c4379: - sv_function_helpervar_b9,r3__condX3
        + 7.8286385936948 sv_function_helpervar_b9,r3__Z3 <= 0
 c4380: - aux_c_b0,r9 + sv_function_helpervar_b0,r9__condX1
        + sv_function_helpervar_b0,r9__condX3
        + sv_function_helpervar_b0,r9__condX2  = 0
 c4381: - 3 aux_cap_helper__b5,r10,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b5,r10,band_HIGH_PAIRED__condX3 <= 0
 c4382: 3.1788778092 aux_cap_helper__b9,r10,band_LOW_PAIRED__condX1
        + 4.4504289328 aux_cap_helper__b9,r10,band_LOW_PAIRED__condX2
        - 1.2715511236 aux_cap_helper__b9,r10,band_LOW_PAIRED__Z2
        - aux_cap_b9,r10,band_LOW_PAIRED  = 0
 c4383: - aux_cap_helper__b1,r12,band_LOW_PAIRED__Z1
        + aux_cap_helper__b1,r12,band_LOW_PAIRED__condX1 <= 0
 c4384: 2 aux_cap_helper__b1,r12,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b1,r12,band_HIGH_PAIRED__condX3 <= 0
 c4385: - 0.158453086893361 sv_function_helpervar_b5,r6__Z1
        + sv_function_helpervar_b5,r6__condX1 <= 0
 c4386: aux_cap_helper__b3,r6,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b3,r6,band_HIGH_PAIRED__condX2 <= 0
 c4387: - 2 aux_cap_helper__b2,r8,band_LOW_PAIRED__Z2
        + aux_cap_helper__b2,r8,band_LOW_PAIRED__condX2 <= 0
 c4388: - 2 aux_cap_helper__b1,r6,band_LOW_PAIRED__Z2
        + aux_cap_helper__b1,r6,band_LOW_PAIRED__condX2 <= 0
 c4389: 4.4504289328 aux_cap_helper__b1,r2,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b1,r2,band_LOW_PAIRED__condX1
        - 1.2715511236 aux_cap_helper__b1,r2,band_LOW_PAIRED__Z2
        - aux_cap_b1,r2,band_LOW_PAIRED  = 0
 c4390: - aux_cap_helper__b6,r4,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b6,r4,band_HIGH_PAIRED__Z3 <= 0
 c4391: aux_cap_helper__b7,r4,band_UNPAIRED__Z2
        + aux_cap_helper__b7,r4,band_UNPAIRED__Z1  = 1
 c4392: - aux_cap_helper__b9,r9,band_UNPAIRED__Z1
        + aux_cap_helper__b9,r9,band_UNPAIRED__condX1 <= 0
 c4393: X__b0,r12,band_UNPAIRED + X__b8,r12,band_UNPAIRED
        + X__b6,r12,band_UNPAIRED + X__b3,r12,band_UNPAIRED
        + X__b9,r12,band_UNPAIRED + X__b2,r12,band_UNPAIRED
        + X__b7,r12,band_UNPAIRED + X__b4,r12,band_UNPAIRED
        + X__b5,r12,band_UNPAIRED + X__b1,r12,band_UNPAIRED <= 2
 c4394: - sv_function_helpervar_b0,r2__condX3
        + 5.87408986601521 sv_function_helpervar_b0,r2__Z3 <= 0
 c4395: - aux_cap_helper__b8,r8,band_HIGH_PAIRED__condX1 <= 0
 c4396: sv_function_helpervar_b3,r12__condX3
        + sv_function_helpervar_b3,r12__condX2
        + sv_function_helpervar_b3,r12__condX1 - aux_c_b3,r12  = 0
 c4397: sv_function_helpervar_b4,r7__condX2
        + sv_function_helpervar_b4,r7__condX3
        + sv_function_helpervar_b4,r7__condX1 - aux_c_b4,r7  = 0
 c4398: sv_function_helpervar_b4,r5__condX2
        - 5.81826105166577 sv_function_helpervar_b4,r5__Z2 <= 0
 c4399: - 2 aux_cap_helper__b9,r0,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b9,r0,band_HIGH_PAIRED__condX2 <= 0
 c4400: X__b9,r11,band_UNPAIRED - GlobalBidder_W_ir__b9,r11
        + X__b9,r11,band_HIGH_PAIRED + X__b9,r11,band_LOW_PAIRED >= 0
 c4401: sv_function_helpervar_b6,r1__condX1
        - 0.158453086915037 sv_function_helpervar_b6,r1__Z1 <= 0
 c4402: - X__b7,r8,band_UNPAIRED + aux_cap_helper__b7,r8,band_UNPAIRED__condX1
        + aux_cap_helper__b7,r8,band_UNPAIRED__condX2  = 0
 c4403: aux_cap_helper__b4,r11,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b4,r11,band_HIGH_PAIRED__Z2 <= 0
 c4404: - 2 aux_cap_helper__b9,r10,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b9,r10,band_HIGH_PAIRED__condX2 <= 0
 c4405: - X__b3,r8,band_LOW_PAIRED
        + aux_cap_helper__b3,r8,band_LOW_PAIRED__condX1
        + aux_cap_helper__b3,r8,band_LOW_PAIRED__condX2  = 0
 c4406: aux_cap_helper__b4,r10,band_UNPAIRED__Z2
        - aux_cap_helper__b4,r10,band_UNPAIRED__condX2 <= 0
 c4407: 1995.2925268505 sv_function_helpervar_b9,r5__condX1
        + 70.7269586605 sv_function_helpervar_b9,r5__condX2
        + 39.1721788004 sv_function_helpervar_b9,r5__condX3
        + 550.268661532844 sv_function_helpervar_b9,r5__Z3
        + 304.953355263228 sv_function_helpervar_b9,r5__Z2 - aux_sv_b9,r5  = 0
 c4408: sv_function_helpervar_b8,r8__condX2
        - 6.47080441264293 sv_function_helpervar_b8,r8__Z2 <= 0
 c4409: aux_cap_helper__b1,r0,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b1,r0,band_UNPAIRED__Z2 <= 0
 c4410: aux_cap_helper__b1,r8,band_UNPAIRED__condX1
        - aux_cap_helper__b1,r8,band_UNPAIRED__Z1 <= 0
 c4411: - 5.78349746755236 sv_function_helpervar_b1,r3__Z2
        + sv_function_helpervar_b1,r3__condX2 <= 0
 c4412: sv_function_helpervar_b7,r2__Z2 + sv_function_helpervar_b7,r2__Z3
        + sv_function_helpervar_b7,r2__Z1  = 1
 c4413: - sv_function_helpervar_b2,r10__condX2
        + 0.158453086903173 sv_function_helpervar_b2,r10__Z2 <= 0
 c4414: 456.283945990949 aux_sv_b2,r10 - aux_Omega_b2,r10  = 0
 c4415: aux_cap_helper__b9,r1,band_LOW_PAIRED__Z1
        + aux_cap_helper__b9,r1,band_LOW_PAIRED__Z2  = 1
 c4416: - aux_cap_helper__b0,r9,band_HIGH_PAIRED__condX1 <= 0
 c4417: - aux_sv_b5,r0 + 978.9018729098 sv_function_helpervar_b5,r0__condX1
        + 15.8629726465 sv_function_helpervar_b5,r0__condX3
        + 44.7238571193 sv_function_helpervar_b5,r0__condX2
        + 323.127870436602 sv_function_helpervar_b5,r0__Z3
        + 148.023390340347 sv_function_helpervar_b5,r0__Z2  = 0
 c4418: - sv_function_helpervar_b7,r6__condX2
        + 0.158453086898302 sv_function_helpervar_b7,r6__Z2 <= 0
 c4419: - aux_cap_helper__b0,r7,band_LOW_PAIRED__condX2
        + aux_cap_helper__b0,r7,band_LOW_PAIRED__Z2 <= 0
 c4420: - 0.158453086929873 sv_function_helpervar_b0,r7__Z1
        + sv_function_helpervar_b0,r7__condX1 <= 0
 c4421: GlobalBidder_PSI__i9,k1 - 20668026.8748422 GlobalBidder_hat{W}__i9,k1
        <= 0
 c4422: aux_cap_helper__b1,r7,band_UNPAIRED__condX1
        + aux_cap_helper__b1,r7,band_UNPAIRED__condX2 - X__b1,r7,band_UNPAIRED
         = 0
 c4423: aux_cap_helper__b8,r8,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b8,r8,band_HIGH_PAIRED__condX2 <= 0
 c4424: aux_cap_helper__b4,r3,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b4,r3,band_LOW_PAIRED__Z2 <= 0
 c4425: - 2 aux_cap_helper__b5,r5,band_LOW_PAIRED__Z2
        + aux_cap_helper__b5,r5,band_LOW_PAIRED__condX2 <= 0
 c4426: 6.21227438384747 sv_function_helpervar_b3,r2__Z3
        - sv_function_helpervar_b3,r2__condX3 <= 0
 c4427: aux_cap_helper__b5,r5,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b5,r5,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b5,r5,band_HIGH_PAIRED__Z1  = 1
 c4428: - aux_cap_helper__b5,r1,band_HIGH_PAIRED__condX1 <= 0
 c4429: - 0.158453086960565 sv_function_helpervar_b5,r3__Z1
        + sv_function_helpervar_b5,r3__condX1 <= 0
 c4430: - X__b7,r12,band_HIGH_PAIRED
        + aux_cap_helper__b7,r12,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b7,r12,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b7,r12,band_HIGH_PAIRED__condX2  = 0
 c4431: aux_cap_helper__b3,r3,band_LOW_PAIRED__Z1
        + aux_cap_helper__b3,r3,band_LOW_PAIRED__Z2  = 1
 c4432: - 0.253841888 aux_cap_helper__b4,r9,band_UNPAIRED__Z2
        - aux_cap_b4,r9,band_UNPAIRED
        + 0.63460472 aux_cap_helper__b4,r9,band_UNPAIRED__condX1
        + 0.888446608 aux_cap_helper__b4,r9,band_UNPAIRED__condX2  = 0
 c4433: aux_cap_helper__b7,r5,band_LOW_PAIRED__condX2
        + aux_cap_helper__b7,r5,band_LOW_PAIRED__condX1
        - X__b7,r5,band_LOW_PAIRED  = 0
 c4434: aux_cap_helper__b7,r10,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b7,r10,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b7,r10,band_HIGH_PAIRED__Z3  = 1
 c4435: - aux_cap_helper__b1,r2,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b1,r2,band_HIGH_PAIRED__Z3 <= 0
 c4436: - sv_function_helpervar_b3,r7__condX3
        + 5.93657899836289 sv_function_helpervar_b3,r7__Z3 <= 0
 c4437: 20668026.8748422 GlobalBidder_hat{W}__i8,k0 - GlobalBidder_PSI__i8,k0
        + aux_Omega_b8,r12 + aux_Omega_b8,r13 + aux_Omega_b8,r10
        + aux_Omega_b8,r11 + aux_Omega_b8,r5 + aux_Omega_b8,r6
        + aux_Omega_b8,r7 + aux_Omega_b8,r8 + aux_Omega_b8,r9 + aux_Omega_b8,r0
        + aux_Omega_b8,r1 + aux_Omega_b8,r2 + aux_Omega_b8,r3 + aux_Omega_b8,r4
        <= 20668026.8748422
 c4438: - aux_cap_helper__b0,r7,band_LOW_PAIRED__condX1 <= 0
 c4439: - aux_cap_helper__b6,r4,band_UNPAIRED__condX1 <= 0
 c4440: - aux_cap_helper__b3,r6,band_HIGH_PAIRED__condX1 <= 0
 c4441: - aux_cap_helper__b4,r7,band_UNPAIRED__condX1 <= 0
 c4442: aux_cap_helper__b0,r12,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b0,r12,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b0,r12,band_HIGH_PAIRED__Z2  = 1
 c4443: - aux_cap_helper__b5,r9,band_LOW_PAIRED__condX1 <= 0
 c4444: aux_cap_helper__b1,r11,band_LOW_PAIRED__Z2
        + aux_cap_helper__b1,r11,band_LOW_PAIRED__Z1  = 1
 c4445: - aux_cap_helper__b8,r13,band_UNPAIRED__condX1 <= 0
 c4446: - sv_function_helpervar_b0,r12__condX3
        + 6.1155275836706 sv_function_helpervar_b0,r12__Z3 <= 0
 c4447: - aux_cap_helper__b2,r10,band_UNPAIRED__Z1
        + aux_cap_helper__b2,r10,band_UNPAIRED__condX1 <= 0
 c4448: - sv_function_helpervar_b4,r4__condX2
        + 0.158453086903718 sv_function_helpervar_b4,r4__Z2 <= 0
 c4449: - 3 aux_cap_helper__b8,r10,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b8,r10,band_HIGH_PAIRED__condX3 <= 0
 c4450: sv_function_helpervar_b2,r11__condX3
        + sv_function_helpervar_b2,r11__condX2
        + sv_function_helpervar_b2,r11__condX1 - aux_c_b2,r11  = 0
 c4451: 42.1358589058 sv_function_helpervar_b4,r5__condX2
        + 13.9600315993 sv_function_helpervar_b4,r5__condX3
        + 883.4028076228 sv_function_helpervar_b4,r5__condX1 - aux_sv_b4,r5
        + 133.301344946929 sv_function_helpervar_b4,r5__Z2
        + 297.235663562791 sv_function_helpervar_b4,r5__Z3  = 0
 c4452: 6.99600389083181 sv_function_helpervar_b7,r8__Z3
        - sv_function_helpervar_b7,r8__condX3 <= 0
 c4453: - sv_function_helpervar_b1,r0__condX1 <= 0
 c4454: - 2 aux_cap_helper__b4,r10,band_LOW_PAIRED__Z2
        + aux_cap_helper__b4,r10,band_LOW_PAIRED__condX2 <= 0
 c4455: - aux_cap_b7,r8,band_UNPAIRED
        + 0.63460472 aux_cap_helper__b7,r8,band_UNPAIRED__condX1
        + 0.888446608 aux_cap_helper__b7,r8,band_UNPAIRED__condX2
        - 0.253841888 aux_cap_helper__b7,r8,band_UNPAIRED__Z2  = 0
 c4456: aux_cap_helper__b3,r5,band_UNPAIRED__condX1
        - aux_cap_helper__b3,r5,band_UNPAIRED__Z1 <= 0
 c4457: sv_function_helpervar_b6,r11__Z3 + sv_function_helpervar_b6,r11__Z2
        + sv_function_helpervar_b6,r11__Z1  = 1
 c4458: - 0.253841888 aux_cap_helper__b1,r4,band_UNPAIRED__Z2
        - aux_cap_b1,r4,band_UNPAIRED
        + 0.63460472 aux_cap_helper__b1,r4,band_UNPAIRED__condX1
        + 0.888446608 aux_cap_helper__b1,r4,band_UNPAIRED__condX2  = 0
 c4459: aux_cap_helper__b7,r13,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b7,r13,band_HIGH_PAIRED__Z1 <= 0
 c4460: aux_cap_helper__b5,r1,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b5,r1,band_HIGH_PAIRED__condX2 <= 0
 c4461: sv_function_helpervar_b6,r2__condX3
        - 15.8453086887446 sv_function_helpervar_b6,r2__Z3 <= 0
 c4462: sv_function_helpervar_b1,r5__Z2 + sv_function_helpervar_b1,r5__Z1
        + sv_function_helpervar_b1,r5__Z3  = 1
 c4463: - GlobalBidder_W_ir__b9,r13
        + 0.0102040816326531 X__b9,r13,band_LOW_PAIRED
        + 0.0102040816326531 X__b9,r13,band_UNPAIRED
        + 0.0102040816326531 X__b9,r13,band_HIGH_PAIRED <= 0
 c4464: - 1.2715511236 aux_cap_helper__b2,r13,band_LOW_PAIRED__Z2
        + 4.4504289328 aux_cap_helper__b2,r13,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b2,r13,band_LOW_PAIRED__condX1
        - aux_cap_b2,r13,band_LOW_PAIRED  = 0
 c4465: - 0.158453086911987 sv_function_helpervar_b4,r3__Z1
        + sv_function_helpervar_b4,r3__condX1 <= 0
 c4466: aux_cap_helper__b2,r4,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b2,r4,band_HIGH_PAIRED__Z1 <= 0
 c4467: 0.0102040816326531 X__b8,r3,band_HIGH_PAIRED
        + 0.0102040816326531 X__b8,r3,band_UNPAIRED
        + 0.0102040816326531 X__b8,r3,band_LOW_PAIRED
        - GlobalBidder_W_ir__b8,r3 <= 0
 c4468: - aux_cap_helper__b0,r10,band_LOW_PAIRED__Z1
        + aux_cap_helper__b0,r10,band_LOW_PAIRED__condX1 <= 0
 c4469: aux_cap_helper__b3,r5,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b3,r5,band_HIGH_PAIRED__Z1 <= 0
 c4470: sv_function_helpervar_b2,r10__condX2
        - 5.98243245052792 sv_function_helpervar_b2,r10__Z2 <= 0
 c4471: aux_cap_helper__b4,r3,band_LOW_PAIRED__Z2
        + aux_cap_helper__b4,r3,band_LOW_PAIRED__Z1  = 1
 c4472: aux_cap_helper__b6,r4,band_UNPAIRED__Z2
        - aux_cap_helper__b6,r4,band_UNPAIRED__condX2 <= 0
 c4473: - aux_cap_helper__b8,r11,band_UNPAIRED__condX1 <= 0
 c4474: aux_cap_helper__b6,r10,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b6,r10,band_HIGH_PAIRED__Z2 <= 0
 c4475: - aux_cap_helper__b8,r13,band_UNPAIRED__condX2
        + aux_cap_helper__b8,r13,band_UNPAIRED__Z2 <= 0
 c4476: - sv_function_helpervar_b9,r7__condX1 <= 0
 c4477: - aux_cap_helper__b6,r2,band_UNPAIRED__Z1
        + aux_cap_helper__b6,r2,band_UNPAIRED__condX1 <= 0
 c4478: - aux_cap_b9,r2,band_UNPAIRED
        - 0.253841888 aux_cap_helper__b9,r2,band_UNPAIRED__Z2
        + 0.888446608 aux_cap_helper__b9,r2,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b9,r2,band_UNPAIRED__condX1  = 0
 c4479: - aux_cap_helper__b1,r3,band_UNPAIRED__condX1 <= 0
 c4480: aux_cap_helper__b1,r5,band_UNPAIRED__Z2
        - aux_cap_helper__b1,r5,band_UNPAIRED__condX2 <= 0
 c4481: - aux_cap_helper__b3,r2,band_UNPAIRED__condX2
        + aux_cap_helper__b3,r2,band_UNPAIRED__Z2 <= 0
 c4482: - 2 aux_cap_helper__b3,r11,band_LOW_PAIRED__Z2
        + aux_cap_helper__b3,r11,band_LOW_PAIRED__condX2 <= 0
 c4483: - sv_function_helpervar_b3,r2__condX1 <= 0
 c4484: - aux_cap_helper__b8,r6,band_LOW_PAIRED__condX1 <= 0
 c4485: - aux_cap_b3,r12,band_LOW_PAIRED
        + 3.1788778092 aux_cap_helper__b3,r12,band_LOW_PAIRED__condX1
        + 4.4504289328 aux_cap_helper__b3,r12,band_LOW_PAIRED__condX2
        - 1.2715511236 aux_cap_helper__b3,r12,band_LOW_PAIRED__Z2  = 0
 c4486: - aux_cap_helper__b6,r8,band_LOW_PAIRED__condX2
        + aux_cap_helper__b6,r8,band_LOW_PAIRED__Z2 <= 0
 c4487: aux_cap_helper__b7,r12,band_LOW_PAIRED__condX1
        + aux_cap_helper__b7,r12,band_LOW_PAIRED__condX2
        - X__b7,r12,band_LOW_PAIRED  = 0
 c4488: - aux_cap_helper__b3,r8,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b3,r8,band_HIGH_PAIRED__condX1 <= 0
 c4489: - aux_cap_helper__b2,r11,band_LOW_PAIRED__Z1
        + aux_cap_helper__b2,r11,band_LOW_PAIRED__condX1 <= 0
 c4490: aux_cap_helper__b3,r0,band_UNPAIRED__condX1
        - aux_cap_helper__b3,r0,band_UNPAIRED__Z1 <= 0
 c4491: 7.52981976069524 sv_function_helpervar_b7,r4__Z3
        - sv_function_helpervar_b7,r4__condX3 <= 0
 c4492: - X__b9,r2,band_HIGH_PAIRED
        + aux_cap_helper__b9,r2,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b9,r2,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b9,r2,band_HIGH_PAIRED__condX3  = 0
 c4493: - aux_cap_helper__b0,r0,band_UNPAIRED__condX2
        + aux_cap_helper__b0,r0,band_UNPAIRED__Z2 <= 0
 c4494: - aux_cap_helper__b1,r5,band_UNPAIRED__condX1 <= 0
 c4495: - aux_cap_helper__b4,r10,band_HIGH_PAIRED__condX1 <= 0
 c4496: sv_function_helpervar_b1,r1__Z3 + sv_function_helpervar_b1,r1__Z1
        + sv_function_helpervar_b1,r1__Z2  = 1
 c4497: aux_cap_helper__b5,r12,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b5,r12,band_LOW_PAIRED__Z2 <= 0
 c4498: - aux_cap_helper__b6,r8,band_LOW_PAIRED__condX1 <= 0
 c4499: sv_function_helpervar_b7,r2__condX1
        + sv_function_helpervar_b7,r2__condX3
        + sv_function_helpervar_b7,r2__condX2 - aux_c_b7,r2  = 0
 c4500: aux_cap_helper__b9,r6,band_UNPAIRED__Z2
        + aux_cap_helper__b9,r6,band_UNPAIRED__Z1  = 1
 c4501: aux_cap_helper__b2,r8,band_UNPAIRED__Z2
        - aux_cap_helper__b2,r8,band_UNPAIRED__condX2 <= 0
 c4502: X__b9,r5,band_UNPAIRED + X__b2,r5,band_UNPAIRED
        + X__b8,r5,band_UNPAIRED + X__b7,r5,band_UNPAIRED
        + X__b3,r5,band_UNPAIRED + X__b5,r5,band_UNPAIRED
        + X__b6,r5,band_UNPAIRED + X__b4,r5,band_UNPAIRED
        + X__b1,r5,band_UNPAIRED + X__b0,r5,band_UNPAIRED <= 2
 c4503: - aux_cap_helper__b0,r0,band_UNPAIRED__condX1 <= 0
 c4504: - aux_cap_helper__b8,r11,band_UNPAIRED__condX2
        + aux_cap_helper__b8,r11,band_UNPAIRED__Z2 <= 0
 c4505: - sv_function_helpervar_b8,r2__condX3
        + 6.45410399062127 sv_function_helpervar_b8,r2__Z3 <= 0
 c4506: - sv_function_helpervar_b5,r8__condX3
        + 5.7076421829379 sv_function_helpervar_b5,r8__Z3 <= 0
 c4507: 0.0102040816326531 X__b8,r2,band_UNPAIRED
        + 0.0102040816326531 X__b8,r2,band_LOW_PAIRED
        - GlobalBidder_W_ir__b8,r2
        + 0.0102040816326531 X__b8,r2,band_HIGH_PAIRED <= 0
 c4508: - 15.8453086887446 sv_function_helpervar_b0,r6__Z3
        + sv_function_helpervar_b0,r6__condX3 <= 0
 c4509: - sv_function_helpervar_b1,r8__condX1 <= 0
 c4510: - 3 aux_cap_helper__b6,r8,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b6,r8,band_HIGH_PAIRED__condX3 <= 0
 c4511: - X__b2,r5,band_UNPAIRED + aux_cap_helper__b2,r5,band_UNPAIRED__condX2
        + aux_cap_helper__b2,r5,band_UNPAIRED__condX1  = 0
 c4512: - 3 aux_cap_helper__b8,r13,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b8,r13,band_HIGH_PAIRED__condX3 <= 0
 c4513: X__b9,r12,band_LOW_PAIRED + X__b2,r12,band_LOW_PAIRED
        + X__b1,r12,band_LOW_PAIRED + X__b0,r12,band_LOW_PAIRED
        + X__b4,r12,band_LOW_PAIRED + X__b3,r12,band_LOW_PAIRED
        + X__b8,r12,band_LOW_PAIRED + X__b5,r12,band_LOW_PAIRED
        + X__b6,r12,band_LOW_PAIRED + X__b7,r12,band_LOW_PAIRED <= 2
 c4514: - 15.8453086887446 sv_function_helpervar_b3,r5__Z3
        + sv_function_helpervar_b3,r5__condX3 <= 0
 c4515: - X__b9,r10,band_LOW_PAIRED
        + aux_cap_helper__b9,r10,band_LOW_PAIRED__condX1
        + aux_cap_helper__b9,r10,band_LOW_PAIRED__condX2  = 0
 c4516: - aux_cap_helper__b1,r3,band_UNPAIRED__condX2
        + aux_cap_helper__b1,r3,band_UNPAIRED__Z2 <= 0
 c4517: aux_cap_helper__b9,r9,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b9,r9,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b9,r9,band_HIGH_PAIRED__Z1  = 1
 c4518: - aux_cap_helper__b2,r8,band_UNPAIRED__condX1 <= 0
 c4519: X__b8,r0,band_LOW_PAIRED + X__b8,r0,band_HIGH_PAIRED
        + X__b8,r0,band_UNPAIRED - GlobalBidder_W_ir__b8,r0 >= 0
 c4520: aux_cap_helper__b0,r6,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b0,r6,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b0,r6,band_HIGH_PAIRED__Z1  = 1
 c4521: sv_function_helpervar_b7,r3__condX1
        - 0.158453086899856 sv_function_helpervar_b7,r3__Z1 <= 0
 c4522: - 2 aux_cap_helper__b6,r11,band_LOW_PAIRED__Z2
        + aux_cap_helper__b6,r11,band_LOW_PAIRED__condX2 <= 0
 c4523: - X__b5,r12,band_HIGH_PAIRED
        + aux_cap_helper__b5,r12,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b5,r12,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b5,r12,band_HIGH_PAIRED__condX1  = 0
 c4524: - 15.8453086887446 sv_function_helpervar_b7,r0__Z3
        + sv_function_helpervar_b7,r0__condX3 <= 0
 c4525: aux_cap_helper__b6,r10,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b6,r10,band_UNPAIRED__Z2 <= 0
 c4526: aux_cap_helper__b7,r1,band_LOW_PAIRED__Z2
        + aux_cap_helper__b7,r1,band_LOW_PAIRED__Z1  = 1
 c4527: 5.68383285585204 sv_function_helpervar_b5,r11__Z3
        - sv_function_helpervar_b5,r11__condX3 <= 0
 c4528: sv_function_helpervar_b6,r5__Z3 + sv_function_helpervar_b6,r5__Z2
        + sv_function_helpervar_b6,r5__Z1  = 1
 c4529: - aux_cap_helper__b4,r13,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b4,r13,band_HIGH_PAIRED__condX1 <= 0
 c4530: aux_cap_helper__b4,r10,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b4,r10,band_HIGH_PAIRED__condX2 <= 0
 c4531: - sv_function_helpervar_b6,r6__condX3
        + 5.93043276387834 sv_function_helpervar_b6,r6__Z3 <= 0
 c4532: X__b8,r5,band_HIGH_PAIRED + X__b8,r5,band_LOW_PAIRED
        + X__b8,r5,band_UNPAIRED - GlobalBidder_W_ir__b8,r5 >= 0
 c4533: aux_cap_helper__b5,r2,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b5,r2,band_LOW_PAIRED__Z2 <= 0
 c4534: - aux_cap_helper__b9,r5,band_LOW_PAIRED__condX2
        + aux_cap_helper__b9,r5,band_LOW_PAIRED__Z2 <= 0
 c4535: - aux_Omega_b6,r10 + 463.961307685493 aux_sv_b6,r10  = 0
 c4536: - sv_function_helpervar_b4,r10__condX1 <= 0
 c4537: aux_cap_helper__b8,r1,band_UNPAIRED__condX1
        - aux_cap_helper__b8,r1,band_UNPAIRED__Z1 <= 0
 c4538: - aux_cap_b8,r1,band_LOW_PAIRED
        + 4.4504289328 aux_cap_helper__b8,r1,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b8,r1,band_LOW_PAIRED__condX1
        - 1.2715511236 aux_cap_helper__b8,r1,band_LOW_PAIRED__Z2  = 0
 c4539: - X__b1,r5,band_HIGH_PAIRED
        + aux_cap_helper__b1,r5,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b1,r5,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b1,r5,band_HIGH_PAIRED__condX1  = 0
 c4540: - aux_cap_helper__b9,r5,band_LOW_PAIRED__condX1 <= 0
 c4541: sv_function_helpervar_b0,r8__Z3 + sv_function_helpervar_b0,r8__Z1
        + sv_function_helpervar_b0,r8__Z2  = 1
 c4542: aux_cap_helper__b4,r5,band_UNPAIRED__Z2
        - aux_cap_helper__b4,r5,band_UNPAIRED__condX2 <= 0
 c4543: - 1.2715511236 aux_cap_helper__b3,r6,band_LOW_PAIRED__Z2
        - aux_cap_b3,r6,band_LOW_PAIRED
        + 3.1788778092 aux_cap_helper__b3,r6,band_LOW_PAIRED__condX1
        + 4.4504289328 aux_cap_helper__b3,r6,band_LOW_PAIRED__condX2  = 0
 c4544: - aux_cap_helper__b6,r9,band_UNPAIRED__Z1
        + aux_cap_helper__b6,r9,band_UNPAIRED__condX1 <= 0
 c4545: sv_function_helpervar_b1,r2__condX1
        - 0.158453086927277 sv_function_helpervar_b1,r2__Z1 <= 0
 c4546: - X__b6,r2,band_LOW_PAIRED
        + aux_cap_helper__b6,r2,band_LOW_PAIRED__condX2
        + aux_cap_helper__b6,r2,band_LOW_PAIRED__condX1  = 0
 c4547: aux_cap_helper__b3,r10,band_LOW_PAIRED__condX1
        - aux_cap_helper__b3,r10,band_LOW_PAIRED__Z1 <= 0
 c4548: aux_cap_helper__b9,r10,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b9,r10,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b9,r10,band_HIGH_PAIRED__Z1  = 1
 c4549: - sv_function_helpervar_b3,r9__condX2
        + 0.158453086901293 sv_function_helpervar_b3,r9__Z2 <= 0
 c4550: - aux_cap_helper__b4,r5,band_UNPAIRED__condX1 <= 0
 c4551: aux_cap_helper__b6,r7,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b6,r7,band_LOW_PAIRED__Z2 <= 0
 c4552: aux_cap_helper__b3,r7,band_UNPAIRED__condX1
        - aux_cap_helper__b3,r7,band_UNPAIRED__Z1 <= 0
 c4553: 1.21690011250031 aux_cap_helper__b8,r10,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b8,r10,band_HIGH_PAIRED__Z2
        + 1.8253501688 aux_cap_helper__b8,r10,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b8,r10,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b8,r10,band_HIGH_PAIRED__condX1
        - aux_cap_b8,r10,band_HIGH_PAIRED  = 0
 c4554: aux_cap_helper__b1,r12,band_UNPAIRED__condX2
        + aux_cap_helper__b1,r12,band_UNPAIRED__condX1
        - X__b1,r12,band_UNPAIRED  = 0
 c4555: - aux_cap_helper__b6,r2,band_HIGH_PAIRED__condX1 <= 0
 c4556: - sv_function_helpervar_b9,r6__condX2
        + 0.158453086924069 sv_function_helpervar_b9,r6__Z2 <= 0
 c4557: - aux_cap_helper__b2,r3,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b2,r3,band_HIGH_PAIRED__Z3 <= 0
 c4558: aux_cap_helper__b4,r13,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b4,r13,band_LOW_PAIRED__Z2 <= 0
 c4559: sv_function_helpervar_b9,r12__Z3 + sv_function_helpervar_b9,r12__Z1
        + sv_function_helpervar_b9,r12__Z2  = 1
 c4560: aux_cap_helper__b2,r7,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b2,r7,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b2,r7,band_HIGH_PAIRED__condX3
        - X__b2,r7,band_HIGH_PAIRED  = 0
 c4561: 6.19715991137678 sv_function_helpervar_b2,r6__Z3
        - sv_function_helpervar_b2,r6__condX3 <= 0
 c4562: - 6.27092287796977 sv_function_helpervar_b9,r12__Z2
        + sv_function_helpervar_b9,r12__condX2 <= 0
 c4563: - 2 aux_cap_helper__b3,r7,band_LOW_PAIRED__Z2
        + aux_cap_helper__b3,r7,band_LOW_PAIRED__condX2 <= 0
 c4564: - aux_cap_helper__b4,r7,band_HIGH_PAIRED__condX1 <= 0
 c4565: sv_function_helpervar_b0,r2__condX1
        - 0.158453086904217 sv_function_helpervar_b0,r2__Z1 <= 0
 c4566: - aux_cap_b5,r0,band_HIGH_PAIRED
        - 0.8112667416 aux_cap_helper__b5,r0,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b5,r0,band_HIGH_PAIRED__Z3
        + 2.8394335959 aux_cap_helper__b5,r0,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b5,r0,band_HIGH_PAIRED__condX1
        + 1.8253501688 aux_cap_helper__b5,r0,band_HIGH_PAIRED__condX3  = 0
 c4567: 1206.00712896099 aux_sv_b8,r12 - aux_Omega_b8,r12  = 0
 c4568: 7.91255019037666 sv_function_helpervar_b7,r10__Z3
        - sv_function_helpervar_b7,r10__condX3 <= 0
 c4569: - 2 aux_cap_helper__b2,r5,band_LOW_PAIRED__Z2
        + aux_cap_helper__b2,r5,band_LOW_PAIRED__condX2 <= 0
 c4570: sv_function_helpervar_b7,r13__condX1
        - 0.158453086902532 sv_function_helpervar_b7,r13__Z1 <= 0
 c4571: - X__b2,r6,band_LOW_PAIRED
        + aux_cap_helper__b2,r6,band_LOW_PAIRED__condX2
        + aux_cap_helper__b2,r6,band_LOW_PAIRED__condX1  = 0
 c4572: aux_cap_helper__b2,r10,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b2,r10,band_HIGH_PAIRED__Z3 <= 0
 c4573: - aux_cap_helper__b6,r7,band_UNPAIRED__Z1
        + aux_cap_helper__b6,r7,band_UNPAIRED__condX1 <= 0
 c4574: aux_cap_helper__b1,r3,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b1,r3,band_LOW_PAIRED__Z2 <= 0
 c4575: aux_cap_helper__b0,r0,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b0,r0,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b0,r0,band_HIGH_PAIRED__condX1
        - X__b0,r0,band_HIGH_PAIRED  = 0
 c4576: X__b5,r7,band_HIGH_PAIRED + X__b2,r7,band_HIGH_PAIRED
        + X__b1,r7,band_HIGH_PAIRED + X__b0,r7,band_HIGH_PAIRED
        + X__b3,r7,band_HIGH_PAIRED + X__b4,r7,band_HIGH_PAIRED
        + X__b6,r7,band_HIGH_PAIRED + X__b9,r7,band_HIGH_PAIRED
        + X__b7,r7,band_HIGH_PAIRED + X__b8,r7,band_HIGH_PAIRED <= 3
 c4577: sv_function_helpervar_b6,r9__Z3 + sv_function_helpervar_b6,r9__Z2
        + sv_function_helpervar_b6,r9__Z1  = 1
 c4578: - 3 aux_cap_helper__b2,r0,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b2,r0,band_HIGH_PAIRED__condX3 <= 0
 c4579: - 3 aux_cap_helper__b9,r8,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b9,r8,band_HIGH_PAIRED__condX3 <= 0
 c4580: aux_cap_helper__b0,r8,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b0,r8,band_HIGH_PAIRED__Z1 <= 0
 c4581: aux_cap_helper__b9,r2,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b9,r2,band_HIGH_PAIRED__condX2 <= 0
 c4582: sv_function_helpervar_b9,r9__condX1
        + sv_function_helpervar_b9,r9__condX2
        + sv_function_helpervar_b9,r9__condX3 - aux_c_b9,r9  = 0
 c4583: 2.0281668543 aux_cap_helper__b9,r1,band_HIGH_PAIRED__condX1
        + 2.8394335959 aux_cap_helper__b9,r1,band_HIGH_PAIRED__condX2
        + 1.8253501688 aux_cap_helper__b9,r1,band_HIGH_PAIRED__condX3
        - aux_cap_b9,r1,band_HIGH_PAIRED
        + 1.21690011250031 aux_cap_helper__b9,r1,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b9,r1,band_HIGH_PAIRED__Z2  = 0
 c4584: aux_cap_helper__b5,r11,band_UNPAIRED__condX1
        - aux_cap_helper__b5,r11,band_UNPAIRED__Z1 <= 0
 c4585: aux_cap_helper__b6,r13,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b6,r13,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b6,r13,band_HIGH_PAIRED__Z3  = 1
 c4586: aux_cap_helper__b8,r6,band_UNPAIRED__condX1
        - aux_cap_helper__b8,r6,band_UNPAIRED__Z1 <= 0
 c4587: - aux_cap_helper__b6,r5,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b6,r5,band_HIGH_PAIRED__condX1 <= 0
 c4588: - aux_cap_b9,r7,band_LOW_PAIRED
        - 1.2715511236 aux_cap_helper__b9,r7,band_LOW_PAIRED__Z2
        + 3.1788778092 aux_cap_helper__b9,r7,band_LOW_PAIRED__condX1
        + 4.4504289328 aux_cap_helper__b9,r7,band_LOW_PAIRED__condX2  = 0
 c4589: - aux_cap_helper__b6,r2,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b6,r2,band_HIGH_PAIRED__Z2 <= 0
 c4590: - 2 aux_cap_helper__b2,r12,band_LOW_PAIRED__Z2
        + aux_cap_helper__b2,r12,band_LOW_PAIRED__condX2 <= 0
 c4591: - aux_cap_helper__b2,r9,band_LOW_PAIRED__Z1
        + aux_cap_helper__b2,r9,band_LOW_PAIRED__condX1 <= 0
 c4592: sv_function_helpervar_b4,r0__condX2
        + sv_function_helpervar_b4,r0__condX1
        + sv_function_helpervar_b4,r0__condX3 - aux_c_b4,r0  = 0
 c4593: 2 aux_cap_helper__b2,r13,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b2,r13,band_HIGH_PAIRED__condX3 <= 0
 c4594: 1.21690011250031 aux_cap_helper__b2,r0,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b2,r0,band_HIGH_PAIRED__Z2
        + 2.0281668543 aux_cap_helper__b2,r0,band_HIGH_PAIRED__condX1
        + 1.8253501688 aux_cap_helper__b2,r0,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b2,r0,band_HIGH_PAIRED__condX2
        - aux_cap_b2,r0,band_HIGH_PAIRED  = 0
 c4595: - aux_cap_helper__b7,r12,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b7,r12,band_HIGH_PAIRED__Z3 <= 0
 c4596: - X__b3,r0,band_HIGH_PAIRED
        + aux_cap_helper__b3,r0,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b3,r0,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b3,r0,band_HIGH_PAIRED__condX1  = 0
 c4597: 2 aux_cap_helper__b5,r3,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b5,r3,band_HIGH_PAIRED__condX3 <= 0
 c4598: sv_function_helpervar_b8,r4__condX3
        - 15.8453086887446 sv_function_helpervar_b8,r4__Z3 <= 0
 c4599: 4.4504289328 aux_cap_helper__b0,r8,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b0,r8,band_LOW_PAIRED__condX1
        - aux_cap_b0,r8,band_LOW_PAIRED
        - 1.2715511236 aux_cap_helper__b0,r8,band_LOW_PAIRED__Z2  = 0
 c4600: - aux_cap_helper__b0,r12,band_UNPAIRED__Z1
        + aux_cap_helper__b0,r12,band_UNPAIRED__condX1 <= 0
 c4601: aux_cap_helper__b5,r1,band_UNPAIRED__condX2
        + aux_cap_helper__b5,r1,band_UNPAIRED__condX1 - X__b5,r1,band_UNPAIRED
         = 0
 c4602: aux_cap_helper__b4,r7,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b4,r7,band_HIGH_PAIRED__condX2 <= 0
 c4603: - 5.87988364123941 sv_function_helpervar_b5,r12__Z2
        + sv_function_helpervar_b5,r12__condX2 <= 0
 c4604: - aux_cap_helper__b7,r7,band_HIGH_PAIRED__condX1 <= 0
 c4605: aux_cap_helper__b8,r6,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b8,r6,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b8,r6,band_HIGH_PAIRED__Z2  = 1
 c4606: 0.15845308693396 sv_function_helpervar_b7,r4__Z2
        - sv_function_helpervar_b7,r4__condX2 <= 0
 c4607: aux_cap_helper__b8,r0,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b8,r0,band_HIGH_PAIRED__Z3 <= 0
 c4608: - 2 aux_cap_helper__b7,r4,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b7,r4,band_HIGH_PAIRED__condX2 <= 0
 c4609: - 5.45154056009326 sv_function_helpervar_b4,r10__Z2
        + sv_function_helpervar_b4,r10__condX2 <= 0
 c4610: - 2 aux_cap_helper__b4,r0,band_LOW_PAIRED__Z2
        + aux_cap_helper__b4,r0,band_LOW_PAIRED__condX2 <= 0
 c4611: - sv_function_helpervar_b4,r4__condX3
        + 6.21753278941992 sv_function_helpervar_b4,r4__Z3 <= 0
 c4612: - X__b4,r3,band_UNPAIRED + aux_cap_helper__b4,r3,band_UNPAIRED__condX1
        + aux_cap_helper__b4,r3,band_UNPAIRED__condX2  = 0
 c4613: aux_cap_helper__b2,r2,band_UNPAIRED__condX1
        + aux_cap_helper__b2,r2,band_UNPAIRED__condX2 - X__b2,r2,band_UNPAIRED
         = 0
 c4614: - aux_cap_helper__b3,r2,band_UNPAIRED__condX1 <= 0
 c4615: - sv_function_helpervar_b0,r5__condX2
        + 0.158453086893356 sv_function_helpervar_b0,r5__Z2 <= 0
 c4616: aux_cap_helper__b6,r6,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b6,r6,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b6,r6,band_HIGH_PAIRED__Z2  = 1
 c4617: - aux_cap_helper__b2,r6,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b2,r6,band_HIGH_PAIRED__Z3 <= 0
 c4618: aux_cap_helper__b8,r3,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b8,r3,band_HIGH_PAIRED__Z3 <= 0
 c4619: aux_cap_helper__b0,r3,band_LOW_PAIRED__Z1
        + aux_cap_helper__b0,r3,band_LOW_PAIRED__Z2  = 1
 c4620: - X__b1,r10,band_HIGH_PAIRED
        + aux_cap_helper__b1,r10,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b1,r10,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b1,r10,band_HIGH_PAIRED__condX1  = 0
 c4621: - aux_cap_helper__b5,r6,band_LOW_PAIRED__condX1 <= 0
 c4622: sv_function_helpervar_b8,r6__Z3 + sv_function_helpervar_b8,r6__Z2
        + sv_function_helpervar_b8,r6__Z1  = 1
 c4623: aux_cap_helper__b3,r13,band_LOW_PAIRED__Z2
        + aux_cap_helper__b3,r13,band_LOW_PAIRED__Z1  = 1
 c4624: - sv_function_helpervar_b4,r11__condX1 <= 0
 c4625: aux_cap_helper__b8,r5,band_UNPAIRED__Z2
        + aux_cap_helper__b8,r5,band_UNPAIRED__Z1  = 1
 c4626: - aux_cap_helper__b7,r7,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b7,r7,band_HIGH_PAIRED__Z2 <= 0
 c4627: aux_cap_b4,r4,band_UNPAIRED + aux_cap_b4,r4,band_LOW_PAIRED
        - aux_c_b4,r4 + aux_cap_b4,r4,band_HIGH_PAIRED  = 0
 c4628: 0.63460472 aux_cap_helper__b0,r4,band_UNPAIRED__condX1
        + 0.888446608 aux_cap_helper__b0,r4,band_UNPAIRED__condX2
        - aux_cap_b0,r4,band_UNPAIRED
        - 0.253841888 aux_cap_helper__b0,r4,band_UNPAIRED__Z2  = 0
 c4629: - aux_cap_helper__b6,r5,band_LOW_PAIRED__condX1 <= 0
 c4630: aux_cap_helper__b0,r5,band_LOW_PAIRED__Z1
        + aux_cap_helper__b0,r5,band_LOW_PAIRED__Z2  = 1
 c4631: - aux_cap_helper__b9,r9,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b9,r9,band_HIGH_PAIRED__Z2 <= 0
 c4632: - 0.253841888 aux_cap_helper__b8,r12,band_UNPAIRED__Z2
        + 0.888446608 aux_cap_helper__b8,r12,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b8,r12,band_UNPAIRED__condX1
        - aux_cap_b8,r12,band_UNPAIRED  = 0
 c4633: - aux_cap_b1,r1,band_UNPAIRED
        - 0.253841888 aux_cap_helper__b1,r1,band_UNPAIRED__Z2
        + 0.63460472 aux_cap_helper__b1,r1,band_UNPAIRED__condX1
        + 0.888446608 aux_cap_helper__b1,r1,band_UNPAIRED__condX2  = 0
 c4634: sv_function_helpervar_b2,r8__condX3
        - 15.8453086887446 sv_function_helpervar_b2,r8__Z3 <= 0
 c4635: - X__b4,r5,band_HIGH_PAIRED
        + aux_cap_helper__b4,r5,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b4,r5,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b4,r5,band_HIGH_PAIRED__condX2  = 0
 c4636: 2 aux_cap_helper__b8,r6,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b8,r6,band_HIGH_PAIRED__condX3 <= 0
 c4637: 6.25459312389671 sv_function_helpervar_b5,r7__Z3
        - sv_function_helpervar_b5,r7__condX3 <= 0
 c4638: - 1.2715511236 aux_cap_helper__b0,r4,band_LOW_PAIRED__Z2
        + 3.1788778092 aux_cap_helper__b0,r4,band_LOW_PAIRED__condX1
        + 4.4504289328 aux_cap_helper__b0,r4,band_LOW_PAIRED__condX2
        - aux_cap_b0,r4,band_LOW_PAIRED  = 0
 c4639: aux_cap_helper__b2,r1,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b2,r1,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b2,r1,band_HIGH_PAIRED__condX3
        - X__b2,r1,band_HIGH_PAIRED  = 0
 c4640: - aux_cap_helper__b7,r4,band_LOW_PAIRED__condX1 <= 0
 c4641: sv_function_helpervar_b0,r12__condX3
        - 15.8453086887446 sv_function_helpervar_b0,r12__Z3 <= 0
 c4642: aux_cap_b2,r3,band_LOW_PAIRED + aux_cap_b2,r3,band_UNPAIRED
        + aux_cap_b2,r3,band_HIGH_PAIRED - aux_c_b2,r3  = 0
 c4643: 0.0102040816326531 X__b8,r13,band_UNPAIRED
        + 0.0102040816326531 X__b8,r13,band_LOW_PAIRED
        - GlobalBidder_W_ir__b8,r13
        + 0.0102040816326531 X__b8,r13,band_HIGH_PAIRED <= 0
 c4644: - aux_sv_b5,r10 + 978.901872532 sv_function_helpervar_b5,r10__condX1
        + 45.5609133223 sv_function_helpervar_b5,r10__condX2
        + 15.6887954922 sv_function_helpervar_b5,r10__condX3
        + 147.890756198254 sv_function_helpervar_b5,r10__Z2
        + 325.887761213195 sv_function_helpervar_b5,r10__Z3  = 0
 c4645: - aux_sv_b1,r12 + 7.466678556 sv_function_helpervar_b1,r12__condX2
        + 150.9476061465 sv_function_helpervar_b1,r12__condX1
        + 2.3381832911 sv_function_helpervar_b1,r12__condX3
        + 51.5363719636661 sv_function_helpervar_b1,r12__Z3
        + 22.734995888901 sv_function_helpervar_b1,r12__Z2  = 0
 c4646: - aux_cap_helper__b5,r6,band_LOW_PAIRED__condX2
        + aux_cap_helper__b5,r6,band_LOW_PAIRED__Z2 <= 0
 c4647: - aux_cap_b8,r11,band_LOW_PAIRED
        + 3.1788778092 aux_cap_helper__b8,r11,band_LOW_PAIRED__condX1
        + 4.4504289328 aux_cap_helper__b8,r11,band_LOW_PAIRED__condX2
        - 1.2715511236 aux_cap_helper__b8,r11,band_LOW_PAIRED__Z2  = 0
 c4648: - 2 aux_cap_helper__b9,r3,band_UNPAIRED__Z2
        + aux_cap_helper__b9,r3,band_UNPAIRED__condX2 <= 0
 c4649: - 2 aux_cap_helper__b1,r13,band_LOW_PAIRED__Z2
        + aux_cap_helper__b1,r13,band_LOW_PAIRED__condX2 <= 0
 c4650: - sv_function_helpervar_b6,r11__condX1 <= 0
 c4651: - 5.57220532880487 sv_function_helpervar_b0,r9__Z2
        + sv_function_helpervar_b0,r9__condX2 <= 0
 c4652: - sv_function_helpervar_b6,r9__condX1 <= 0
 c4653: sv_function_helpervar_b2,r6__Z1 + sv_function_helpervar_b2,r6__Z2
        + sv_function_helpervar_b2,r6__Z3  = 1
 c4654: aux_cap_helper__b4,r8,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b4,r8,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b4,r8,band_HIGH_PAIRED__Z1  = 1
 c4655: aux_cap_b1,r5,band_UNPAIRED + aux_cap_b1,r5,band_LOW_PAIRED
        + aux_cap_b1,r5,band_HIGH_PAIRED - aux_c_b1,r5  = 0
 c4656: - 15.8453086887446 sv_function_helpervar_b1,r3__Z3
        + sv_function_helpervar_b1,r3__condX3 <= 0
 c4657: - aux_cap_helper__b4,r0,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b4,r0,band_HIGH_PAIRED__Z2 <= 0
 c4658: aux_cap_helper__b8,r6,band_UNPAIRED__condX2
        + aux_cap_helper__b8,r6,band_UNPAIRED__condX1 - X__b8,r6,band_UNPAIRED
         = 0
 c4659: - aux_cap_helper__b4,r0,band_HIGH_PAIRED__condX1 <= 0
 c4660: - sv_function_helpervar_b3,r1__condX1 <= 0
 c4661: sv_function_helpervar_b9,r8__condX3
        + sv_function_helpervar_b9,r8__condX2
        + sv_function_helpervar_b9,r8__condX1 - aux_c_b9,r8  = 0
 c4662: sv_function_helpervar_b9,r13__condX2
        - 6.67287044758352 sv_function_helpervar_b9,r13__Z2 <= 0
 c4663: aux_cap_b1,r13,band_HIGH_PAIRED + aux_cap_b1,r13,band_LOW_PAIRED
        - aux_c_b1,r13 + aux_cap_b1,r13,band_UNPAIRED  = 0
 c4664: aux_cap_helper__b3,r1,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b3,r1,band_LOW_PAIRED__Z2 <= 0
 c4665: aux_cap_helper__b4,r6,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b4,r6,band_HIGH_PAIRED__Z1 <= 0
 c4666: sv_function_helpervar_b9,r3__Z3 + sv_function_helpervar_b9,r3__Z1
        + sv_function_helpervar_b9,r3__Z2  = 1
 c4667: sv_function_helpervar_b9,r11__condX3
        - 15.8453086887446 sv_function_helpervar_b9,r11__Z3 <= 0
 c4668: - GlobalBidder_W_ir__b9,r6 + 0.0102040816326531 X__b9,r6,band_UNPAIRED
        + 0.0102040816326531 X__b9,r6,band_HIGH_PAIRED
        + 0.0102040816326531 X__b9,r6,band_LOW_PAIRED <= 0
 c4669: - sv_function_helpervar_b8,r1__condX1 <= 0
 c4670: - 1.2715511236 aux_cap_helper__b2,r8,band_LOW_PAIRED__Z2
        - aux_cap_b2,r8,band_LOW_PAIRED
        + 3.1788778092 aux_cap_helper__b2,r8,band_LOW_PAIRED__condX1
        + 4.4504289328 aux_cap_helper__b2,r8,band_LOW_PAIRED__condX2  = 0
 c4671: 883.4028075804 sv_function_helpervar_b4,r2__condX1
        + 39.6376016518 sv_function_helpervar_b4,r2__condX2
        + 14.4749955709 sv_function_helpervar_b4,r2__condX3 - aux_sv_b4,r2
        + 289.075900469169 sv_function_helpervar_b4,r2__Z3
        + 133.697201520448 sv_function_helpervar_b4,r2__Z2  = 0
 c4672: aux_cap_helper__b0,r11,band_LOW_PAIRED__Z1
        + aux_cap_helper__b0,r11,band_LOW_PAIRED__Z2  = 1
 c4673: - sv_function_helpervar_b1,r9__condX1 <= 0
 c4674: 2 aux_cap_helper__b3,r4,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b3,r4,band_HIGH_PAIRED__condX3 <= 0
 c4675: - aux_cap_helper__b9,r9,band_HIGH_PAIRED__condX1 <= 0
 c4676: - X__b0,r12,band_HIGH_PAIRED
        + aux_cap_helper__b0,r12,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b0,r12,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b0,r12,band_HIGH_PAIRED__condX3  = 0
 c4677: sv_function_helpervar_b6,r3__condX3
        - 15.8453086887446 sv_function_helpervar_b6,r3__Z3 <= 0
 c4678: 2 aux_cap_helper__b5,r13,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b5,r13,band_HIGH_PAIRED__condX3 <= 0
 c4679: 1.21690011250031 aux_cap_helper__b0,r12,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b0,r12,band_HIGH_PAIRED__Z2
        - aux_cap_b0,r12,band_HIGH_PAIRED
        + 2.8394335959 aux_cap_helper__b0,r12,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b0,r12,band_HIGH_PAIRED__condX1
        + 1.8253501688 aux_cap_helper__b0,r12,band_HIGH_PAIRED__condX3  = 0
 c4680: aux_cap_helper__b6,r11,band_UNPAIRED__condX1
        - aux_cap_helper__b6,r11,band_UNPAIRED__Z1 <= 0
 c4681: aux_cap_helper__b3,r10,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b3,r10,band_HIGH_PAIRED__Z2 <= 0
 c4682: aux_cap_helper__b5,r1,band_LOW_PAIRED__Z2
        + aux_cap_helper__b5,r1,band_LOW_PAIRED__Z1  = 1
 c4683: - sv_function_helpervar_b6,r13__condX2
        + 0.15845308692958 sv_function_helpervar_b6,r13__Z2 <= 0
 c4684: aux_cap_b3,r1,band_LOW_PAIRED + aux_cap_b3,r1,band_HIGH_PAIRED
        - aux_c_b3,r1 + aux_cap_b3,r1,band_UNPAIRED  = 0
 c4685: 6.05183208365148 sv_function_helpervar_b0,r7__Z3
        - sv_function_helpervar_b0,r7__condX3 <= 0
 c4686: - aux_cap_helper__b6,r5,band_LOW_PAIRED__condX2
        + aux_cap_helper__b6,r5,band_LOW_PAIRED__Z2 <= 0
 c4687: - X__b8,r0,band_LOW_PAIRED
        + aux_cap_helper__b8,r0,band_LOW_PAIRED__condX2
        + aux_cap_helper__b8,r0,band_LOW_PAIRED__condX1  = 0
 c4688: - aux_Omega_b4,r9 + 261.95744932233 aux_sv_b4,r9  = 0
 c4689: X__b2,r12,band_HIGH_PAIRED + X__b5,r12,band_HIGH_PAIRED
        + X__b7,r12,band_HIGH_PAIRED + X__b8,r12,band_HIGH_PAIRED
        + X__b4,r12,band_HIGH_PAIRED + X__b0,r12,band_HIGH_PAIRED
        + X__b3,r12,band_HIGH_PAIRED + X__b1,r12,band_HIGH_PAIRED
        + X__b6,r12,band_HIGH_PAIRED + X__b9,r12,band_HIGH_PAIRED <= 3
 c4690: - 15.8453086887446 sv_function_helpervar_b7,r8__Z3
        + sv_function_helpervar_b7,r8__condX3 <= 0
 c4691: 6.28882502139831 sv_function_helpervar_b3,r4__Z3
        - sv_function_helpervar_b3,r4__condX3 <= 0
 c4692: - aux_Omega_b5,r8 + 132.996470192017 aux_sv_b5,r8  = 0
 c4693: - aux_cap_helper__b0,r12,band_LOW_PAIRED__condX2
        + aux_cap_helper__b0,r12,band_LOW_PAIRED__Z2 <= 0
 c4694: - 3 aux_cap_helper__b0,r1,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b0,r1,band_HIGH_PAIRED__condX3 <= 0
 c4695: - 0.158453086938541 sv_function_helpervar_b4,r11__Z1
        + sv_function_helpervar_b4,r11__condX1 <= 0
 c4696: - aux_cap_b9,r4,band_HIGH_PAIRED
        + 2.0281668543 aux_cap_helper__b9,r4,band_HIGH_PAIRED__condX1
        + 2.8394335959 aux_cap_helper__b9,r4,band_HIGH_PAIRED__condX2
        + 1.8253501688 aux_cap_helper__b9,r4,band_HIGH_PAIRED__condX3
        - 0.8112667416 aux_cap_helper__b9,r4,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b9,r4,band_HIGH_PAIRED__Z3  = 0
 c4697: - 2 aux_cap_helper__b2,r3,band_UNPAIRED__Z2
        + aux_cap_helper__b2,r3,band_UNPAIRED__condX2 <= 0
 c4698: - aux_cap_helper__b5,r13,band_LOW_PAIRED__condX2
        + aux_cap_helper__b5,r13,band_LOW_PAIRED__Z2 <= 0
 c4699: aux_cap_helper__b4,r9,band_UNPAIRED__Z1
        + aux_cap_helper__b4,r9,band_UNPAIRED__Z2  = 1
 c4700: - aux_cap_b4,r0,band_HIGH_PAIRED
        + 1.8253501688 aux_cap_helper__b4,r0,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b4,r0,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b4,r0,band_HIGH_PAIRED__condX1
        - 0.8112667416 aux_cap_helper__b4,r0,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b4,r0,band_HIGH_PAIRED__Z3  = 0
 c4701: aux_cap_helper__b7,r3,band_UNPAIRED__Z2
        + aux_cap_helper__b7,r3,band_UNPAIRED__Z1  = 1
 c4702: - aux_cap_helper__b0,r12,band_LOW_PAIRED__condX1 <= 0
 c4703: aux_cap_helper__b7,r2,band_LOW_PAIRED__condX1
        - aux_cap_helper__b7,r2,band_LOW_PAIRED__Z1 <= 0
 c4704: - aux_cap_helper__b0,r5,band_UNPAIRED__Z1
        + aux_cap_helper__b0,r5,band_UNPAIRED__condX1 <= 0
 c4705: aux_cap_helper__b4,r9,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b4,r9,band_LOW_PAIRED__Z2 <= 0
 c4706: aux_cap_helper__b6,r0,band_LOW_PAIRED__condX1
        - aux_cap_helper__b6,r0,band_LOW_PAIRED__Z1 <= 0
 c4707: - aux_cap_helper__b2,r1,band_HIGH_PAIRED__condX1 <= 0
 c4708: sv_function_helpervar_b1,r7__condX2
        - 5.60179427758248 sv_function_helpervar_b1,r7__Z2 <= 0
 c4709: aux_cap_helper__b2,r10,band_UNPAIRED__Z1
        + aux_cap_helper__b2,r10,band_UNPAIRED__Z2  = 1
 c4710: sv_function_helpervar_b0,r2__condX3
        + sv_function_helpervar_b0,r2__condX1
        + sv_function_helpervar_b0,r2__condX2 - aux_c_b0,r2  = 0
 c4711: - sv_function_helpervar_b2,r2__condX2
        + 0.158453086894398 sv_function_helpervar_b2,r2__Z2 <= 0
 c4712: aux_cap_helper__b7,r9,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b7,r9,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b7,r9,band_HIGH_PAIRED__Z3  = 1
 c4713: sv_function_helpervar_b5,r0__Z3 + sv_function_helpervar_b5,r0__Z2
        + sv_function_helpervar_b5,r0__Z1  = 1
 c4714: 4.4504289328 aux_cap_helper__b7,r3,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b7,r3,band_LOW_PAIRED__condX1
        - aux_cap_b7,r3,band_LOW_PAIRED
        - 1.2715511236 aux_cap_helper__b7,r3,band_LOW_PAIRED__Z2  = 0
 c4715: sv_function_helpervar_b2,r2__Z3 + sv_function_helpervar_b2,r2__Z1
        + sv_function_helpervar_b2,r2__Z2  = 1
 c4716: aux_cap_helper__b3,r9,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b3,r9,band_HIGH_PAIRED__condX2 <= 0
 c4717: - aux_c_b7,r13 + sv_function_helpervar_b7,r13__condX3
        + sv_function_helpervar_b7,r13__condX2
        + sv_function_helpervar_b7,r13__condX1  = 0
 c4718: - aux_cap_helper__b8,r3,band_UNPAIRED__condX1 <= 0
 c4719: - 3 aux_cap_helper__b7,r2,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b7,r2,band_HIGH_PAIRED__condX3 <= 0
 c4720: - aux_cap_helper__b1,r5,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b1,r5,band_HIGH_PAIRED__Z3 <= 0
 c4721: - 2 aux_cap_helper__b8,r7,band_UNPAIRED__Z2
        + aux_cap_helper__b8,r7,band_UNPAIRED__condX2 <= 0
 c4722: aux_cap_helper__b9,r12,band_LOW_PAIRED__Z2
        + aux_cap_helper__b9,r12,band_LOW_PAIRED__Z1  = 1
 c4723: - 0.8112667416 aux_cap_helper__b3,r0,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b3,r0,band_HIGH_PAIRED__Z3
        - aux_cap_b3,r0,band_HIGH_PAIRED
        + 1.8253501688 aux_cap_helper__b3,r0,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b3,r0,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b3,r0,band_HIGH_PAIRED__condX1  = 0
 c4724: aux_cap_helper__b3,r9,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b3,r9,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b3,r9,band_HIGH_PAIRED__Z2  = 1
 c4725: - 1.2715511236 aux_cap_helper__b8,r5,band_LOW_PAIRED__Z2
        + 4.4504289328 aux_cap_helper__b8,r5,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b8,r5,band_LOW_PAIRED__condX1
        - aux_cap_b8,r5,band_LOW_PAIRED  = 0
 c4726: - aux_cap_helper__b3,r9,band_HIGH_PAIRED__condX1 <= 0
 c4727: - 0.158453086963932 sv_function_helpervar_b1,r8__Z1
        + sv_function_helpervar_b1,r8__condX1 <= 0
 c4728: aux_cap_helper__b2,r2,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b2,r2,band_LOW_PAIRED__Z2 <= 0
 c4729: sv_function_helpervar_b7,r6__condX2
        - 7.48311679006299 sv_function_helpervar_b7,r6__Z2 <= 0
 c4730: aux_cap_helper__b1,r1,band_UNPAIRED__condX1
        + aux_cap_helper__b1,r1,band_UNPAIRED__condX2 - X__b1,r1,band_UNPAIRED
         = 0
 c4731: - X__b4,r13,band_LOW_PAIRED
        + aux_cap_helper__b4,r13,band_LOW_PAIRED__condX1
        + aux_cap_helper__b4,r13,band_LOW_PAIRED__condX2  = 0
 c4732: aux_cap_helper__b9,r3,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b9,r3,band_HIGH_PAIRED__Z2 <= 0
 c4733: - 0.158453086937505 sv_function_helpervar_b0,r9__Z1
        + sv_function_helpervar_b0,r9__condX1 <= 0
 c4734: aux_cap_helper__b7,r10,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b7,r10,band_UNPAIRED__Z2 <= 0
 c4735: aux_cap_helper__b6,r11,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b6,r11,band_HIGH_PAIRED__Z3 <= 0
 c4736: - aux_cap_helper__b2,r1,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b2,r1,band_HIGH_PAIRED__Z2 <= 0
 c4737: - 2 aux_cap_helper__b9,r13,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b9,r13,band_HIGH_PAIRED__condX2 <= 0
 c4738: - sv_function_helpervar_b2,r8__condX3
        + 5.78536513589384 sv_function_helpervar_b2,r8__Z3 <= 0
 c4739: - aux_cap_helper__b7,r10,band_HIGH_PAIRED__condX1 <= 0
 c4740: - aux_cap_helper__b7,r4,band_LOW_PAIRED__condX2
        + aux_cap_helper__b7,r4,band_LOW_PAIRED__Z2 <= 0
 c4741: - sv_function_helpervar_b3,r7__condX2
        + 0.158453086911699 sv_function_helpervar_b3,r7__Z2 <= 0
 c4742: - aux_cap_helper__b8,r6,band_LOW_PAIRED__condX2
        + aux_cap_helper__b8,r6,band_LOW_PAIRED__Z2 <= 0
 c4743: - aux_cap_helper__b1,r1,band_UNPAIRED__Z1
        + aux_cap_helper__b1,r1,band_UNPAIRED__condX1 <= 0
 c4744: - aux_cap_helper__b5,r8,band_UNPAIRED__condX2
        + aux_cap_helper__b5,r8,band_UNPAIRED__Z2 <= 0
 c4745: 5.74218277061943 sv_function_helpervar_b4,r12__Z3
        - sv_function_helpervar_b4,r12__condX3 <= 0
 c4746: aux_cap_helper__b1,r9,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b1,r9,band_HIGH_PAIRED__Z3 <= 0
 c4747: aux_cap_helper__b1,r7,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b1,r7,band_UNPAIRED__Z2 <= 0
 c4748: - 2 aux_cap_helper__b3,r3,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b3,r3,band_HIGH_PAIRED__condX2 <= 0
 c4749: aux_cap_helper__b2,r4,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b2,r4,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b2,r4,band_HIGH_PAIRED__condX3
        - X__b2,r4,band_HIGH_PAIRED  = 0
 c4750: 53.8308229164 sv_function_helpervar_b3,r7__condX2
        + 1152.1877365471 sv_function_helpervar_b3,r7__condX1
        + 18.4249353109 sv_function_helpervar_b3,r7__condX3
        + 174.038043495591 sv_function_helpervar_b3,r7__Z2
        + 384.227892272793 sv_function_helpervar_b3,r7__Z3 - aux_sv_b3,r7  = 0
 c4751: aux_cap_helper__b2,r7,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b2,r7,band_HIGH_PAIRED__Z1 <= 0
 c4752: - aux_cap_helper__b5,r13,band_LOW_PAIRED__condX1 <= 0
 c4753: - X__b4,r0,band_UNPAIRED + aux_cap_helper__b4,r0,band_UNPAIRED__condX2
        + aux_cap_helper__b4,r0,band_UNPAIRED__condX1  = 0
 c4754: aux_cap_helper__b5,r7,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b5,r7,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b5,r7,band_HIGH_PAIRED__Z3  = 1
 c4755: - aux_sv_b2,r4 + 1.891610566 sv_function_helpervar_b2,r4__condX3
        + 114.7166939416 sv_function_helpervar_b2,r4__condX1
        + 5.0956375629 sv_function_helpervar_b2,r4__condX2
        + 17.3697947724107 sv_function_helpervar_b2,r4__Z2
        + 37.34986249316 sv_function_helpervar_b2,r4__Z3  = 0
 c4756: - sv_function_helpervar_b8,r12__condX1 <= 0
 c4757: - aux_cap_helper__b8,r1,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b8,r1,band_HIGH_PAIRED__Z2 <= 0
 c4758: 7.25480608867014 sv_function_helpervar_b8,r3__Z3
        - sv_function_helpervar_b8,r3__condX3 <= 0
 c4759: aux_cap_helper__b1,r3,band_LOW_PAIRED__Z1
        + aux_cap_helper__b1,r3,band_LOW_PAIRED__Z2  = 1
 c4760: - aux_cap_helper__b3,r11,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b3,r11,band_HIGH_PAIRED__Z3 <= 0
 c4761: - 0.158453086936519 sv_function_helpervar_b9,r1__Z1
        + sv_function_helpervar_b9,r1__condX1 <= 0
 c4762: - 5.83047977299225 sv_function_helpervar_b1,r10__Z2
        + sv_function_helpervar_b1,r10__condX2 <= 0
 c4763: - aux_Omega_b9,r5 + 1278.6486632813 aux_sv_b9,r5  = 0
 c4764: - aux_cap_helper__b7,r10,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b7,r10,band_HIGH_PAIRED__Z2 <= 0
 c4765: aux_cap_helper__b9,r8,band_LOW_PAIRED__Z2
        + aux_cap_helper__b9,r8,band_LOW_PAIRED__Z1  = 1
 c4766: - 2 aux_cap_helper__b1,r0,band_LOW_PAIRED__Z2
        + aux_cap_helper__b1,r0,band_LOW_PAIRED__condX2 <= 0
 c4767: aux_cap_helper__b1,r12,band_UNPAIRED__Z1
        + aux_cap_helper__b1,r12,band_UNPAIRED__Z2  = 1
 c4768: - 0.158453086934665 sv_function_helpervar_b3,r5__Z1
        + sv_function_helpervar_b3,r5__condX1 <= 0
 c4769: - aux_Omega_b5,r5 + 597.806310470164 aux_sv_b5,r5  = 0
 c4770: 1.8253501688 aux_cap_helper__b2,r3,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b2,r3,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b2,r3,band_HIGH_PAIRED__condX1
        - aux_cap_b2,r3,band_HIGH_PAIRED
        + 1.21690011250031 aux_cap_helper__b2,r3,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b2,r3,band_HIGH_PAIRED__Z2  = 0
 c4771: aux_cap_helper__b7,r12,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b7,r12,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b7,r12,band_HIGH_PAIRED__Z1  = 1
 c4772: - 3 aux_cap_helper__b7,r9,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b7,r9,band_HIGH_PAIRED__condX3 <= 0
 c4773: 6.47958443516442 sv_function_helpervar_b1,r0__Z3
        - sv_function_helpervar_b1,r0__condX3 <= 0
 c4774: sv_function_helpervar_b5,r3__condX3
        + sv_function_helpervar_b5,r3__condX1
        + sv_function_helpervar_b5,r3__condX2 - aux_c_b5,r3  = 0
 c4775: - aux_cap_helper__b5,r3,band_LOW_PAIRED__condX1 <= 0
 c4776: aux_cap_helper__b7,r11,band_UNPAIRED__Z1
        + aux_cap_helper__b7,r11,band_UNPAIRED__Z2  = 1
 c4777: - aux_Omega_b4,r7 + 2714.43669274465 aux_sv_b4,r7  = 0
 c4778: aux_cap_helper__b5,r8,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b5,r8,band_HIGH_PAIRED__condX2 <= 0
 c4779: aux_cap_helper__b4,r11,band_LOW_PAIRED__Z2
        - aux_cap_helper__b4,r11,band_LOW_PAIRED__condX2 <= 0
 c4780: - aux_cap_helper__b4,r13,band_UNPAIRED__Z1
        + aux_cap_helper__b4,r13,band_UNPAIRED__condX1 <= 0
 c4781: sv_function_helpervar_b6,r1__Z2 + sv_function_helpervar_b6,r1__Z3
        + sv_function_helpervar_b6,r1__Z1  = 1
 c4782: aux_cap_helper__b2,r2,band_HIGH_PAIRED__condX2
        - 2 aux_cap_helper__b2,r2,band_HIGH_PAIRED__Z2 <= 0
 c4783: - aux_cap_helper__b8,r13,band_LOW_PAIRED__condX1 <= 0
 c4784: - aux_cap_b6,r0,band_UNPAIRED
        - 0.253841888 aux_cap_helper__b6,r0,band_UNPAIRED__Z2
        + 0.888446608 aux_cap_helper__b6,r0,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b6,r0,band_UNPAIRED__condX1  = 0
 c4785: - aux_sv_b0,r1 + 56.1374348868058 sv_function_helpervar_b0,r1__Z3
        + 24.6479528683547 sv_function_helpervar_b0,r1__Z2
        + 163.7425743588 sv_function_helpervar_b0,r1__condX1
        + 2.5216971769 sv_function_helpervar_b0,r1__condX3
        + 8.1889442892 sv_function_helpervar_b0,r1__condX2  = 0
 c4786: aux_cap_helper__b3,r1,band_LOW_PAIRED__condX1
        + aux_cap_helper__b3,r1,band_LOW_PAIRED__condX2
        - X__b3,r1,band_LOW_PAIRED  = 0
 c4787: - aux_c_b3,r4 + sv_function_helpervar_b3,r4__condX1
        + sv_function_helpervar_b3,r4__condX3
        + sv_function_helpervar_b3,r4__condX2  = 0
 c4788: 2 aux_cap_helper__b4,r5,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b4,r5,band_HIGH_PAIRED__condX3 <= 0
 c4789: aux_cap_helper__b9,r1,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b9,r1,band_HIGH_PAIRED__Z3 <= 0
 c4790: - 15.8453086887446 sv_function_helpervar_b2,r13__Z3
        + sv_function_helpervar_b2,r13__condX3 <= 0
 c4791: - aux_cap_helper__b6,r2,band_LOW_PAIRED__condX1 <= 0
 c4792: 1.8253501688 aux_cap_helper__b1,r3,band_HIGH_PAIRED__condX3
        + 2.8394335959 aux_cap_helper__b1,r3,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b1,r3,band_HIGH_PAIRED__condX1
        - aux_cap_b1,r3,band_HIGH_PAIRED
        - 0.8112667416 aux_cap_helper__b1,r3,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b1,r3,band_HIGH_PAIRED__Z3  = 0
 c4793: sv_function_helpervar_b8,r13__Z1 + sv_function_helpervar_b8,r13__Z2
        + sv_function_helpervar_b8,r13__Z3  = 1
 c4794: - aux_cap_helper__b7,r7,band_UNPAIRED__condX2
        + aux_cap_helper__b7,r7,band_UNPAIRED__Z2 <= 0
 c4795: aux_cap_helper__b9,r11,band_HIGH_PAIRED__condX3
        - 3 aux_cap_helper__b9,r11,band_HIGH_PAIRED__Z3 <= 0
 c4796: - aux_cap_helper__b2,r12,band_UNPAIRED__condX1 <= 0
 c4797: - aux_cap_b3,r9,band_UNPAIRED
        + 0.888446608 aux_cap_helper__b3,r9,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b3,r9,band_UNPAIRED__condX1
        - 0.253841888 aux_cap_helper__b3,r9,band_UNPAIRED__Z2  = 0
 c4798: aux_cap_helper__b9,r7,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b9,r7,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b9,r7,band_HIGH_PAIRED__Z1  = 1
 c4799: sv_function_helpervar_b3,r13__condX3
        + sv_function_helpervar_b3,r13__condX2
        + sv_function_helpervar_b3,r13__condX1 - aux_c_b3,r13  = 0
 c4800: - aux_cap_helper__b1,r11,band_LOW_PAIRED__condX1 <= 0
 c4801: sv_function_helpervar_b5,r4__Z3 + sv_function_helpervar_b5,r4__Z2
        + sv_function_helpervar_b5,r4__Z1  = 1
 c4802: - aux_cap_helper__b1,r10,band_UNPAIRED__Z1
        + aux_cap_helper__b1,r10,band_UNPAIRED__condX1 <= 0
 c4803: - sv_function_helpervar_b2,r2__condX3
        + 5.80744827810344 sv_function_helpervar_b2,r2__Z3 <= 0
 c4804: aux_cap_helper__b0,r9,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b0,r9,band_HIGH_PAIRED__condX2 <= 0
 c4805: aux_cap_helper__b5,r1,band_UNPAIRED__Z2
        + aux_cap_helper__b5,r1,band_UNPAIRED__Z1  = 1
 c4806: - sv_function_helpervar_b5,r3__condX1 <= 0
 c4807: aux_cap_helper__b2,r10,band_LOW_PAIRED__Z2
        - aux_cap_helper__b2,r10,band_LOW_PAIRED__condX2 <= 0
 c4808: aux_cap_helper__b7,r7,band_LOW_PAIRED__Z2
        - aux_cap_helper__b7,r7,band_LOW_PAIRED__condX2 <= 0
 c4809: 13 GlobalBidder_hat{W}__i8,k2 + GlobalBidder_W_i_8 >= 13
 c4810: - aux_cap_helper__b7,r7,band_UNPAIRED__condX1 <= 0
 c4811: 0.888446608 aux_cap_helper__b5,r3,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b5,r3,band_UNPAIRED__condX1
        - 0.253841888 aux_cap_helper__b5,r3,band_UNPAIRED__Z2
        - aux_cap_b5,r3,band_UNPAIRED  = 0
 c4812: - 3 aux_cap_helper__b4,r2,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b4,r2,band_HIGH_PAIRED__condX3 <= 0
 c4813: - aux_cap_helper__b9,r6,band_UNPAIRED__condX2
        + aux_cap_helper__b9,r6,band_UNPAIRED__Z2 <= 0
 c4814: - sv_function_helpervar_b4,r7__condX1 <= 0
 c4815: 6.63929351348566 sv_function_helpervar_b2,r12__Z3
        - sv_function_helpervar_b2,r12__condX3 <= 0
 c4816: - sv_function_helpervar_b8,r10__condX3
        + 8.23774537961124 sv_function_helpervar_b8,r10__Z3 <= 0
 c4817: - aux_cap_helper__b5,r3,band_LOW_PAIRED__condX2
        + aux_cap_helper__b5,r3,band_LOW_PAIRED__Z2 <= 0
 c4818: - sv_function_helpervar_b2,r10__condX1 <= 0
 c4819: aux_cap_helper__b8,r13,band_LOW_PAIRED__Z2
        - aux_cap_helper__b8,r13,band_LOW_PAIRED__condX2 <= 0
 c4820: - aux_cap_helper__b9,r4,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b9,r4,band_HIGH_PAIRED__Z3 <= 0
 c4821: aux_cap_b8,r2,band_HIGH_PAIRED + aux_cap_b8,r2,band_UNPAIRED
        - aux_c_b8,r2 + aux_cap_b8,r2,band_LOW_PAIRED  = 0
 c4822: - aux_cap_helper__b8,r7,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b8,r7,band_HIGH_PAIRED__condX1 <= 0
 c4823: - aux_cap_helper__b7,r7,band_LOW_PAIRED__condX1 <= 0
 c4824: aux_cap_helper__b8,r11,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b8,r11,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b8,r11,band_HIGH_PAIRED__Z1  = 1
 c4825: - aux_cap_b2,r1,band_UNPAIRED
        - 0.253841888 aux_cap_helper__b2,r1,band_UNPAIRED__Z2
        + 0.63460472 aux_cap_helper__b2,r1,band_UNPAIRED__condX1
        + 0.888446608 aux_cap_helper__b2,r1,band_UNPAIRED__condX2  = 0
 c4826: - aux_cap_b4,r12,band_UNPAIRED
        - 0.253841888 aux_cap_helper__b4,r12,band_UNPAIRED__Z2
        + 0.63460472 aux_cap_helper__b4,r12,band_UNPAIRED__condX1
        + 0.888446608 aux_cap_helper__b4,r12,band_UNPAIRED__condX2  = 0
 c4827: - 2 aux_cap_helper__b5,r2,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b5,r2,band_HIGH_PAIRED__condX2 <= 0
 c4828: 3.1788778092 aux_cap_helper__b7,r12,band_LOW_PAIRED__condX1
        + 4.4504289328 aux_cap_helper__b7,r12,band_LOW_PAIRED__condX2
        - 1.2715511236 aux_cap_helper__b7,r12,band_LOW_PAIRED__Z2
        - aux_cap_b7,r12,band_LOW_PAIRED  = 0
 c4829: - aux_cap_helper__b9,r6,band_UNPAIRED__condX1 <= 0
 c4830: - 2 aux_cap_helper__b7,r11,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b7,r11,band_HIGH_PAIRED__condX2 <= 0
 c4831: 494.660027243721 sv_function_helpervar_b8,r1__Z3
        + 249.088243025885 sv_function_helpervar_b8,r1__Z2
        + 64.357604509 sv_function_helpervar_b8,r1__condX2
        + 29.3877604635 sv_function_helpervar_b8,r1__condX3
        + 1636.3575428335 sv_function_helpervar_b8,r1__condX1 - aux_sv_b8,r1
         = 0
 c4832: - aux_cap_helper__b0,r11,band_HIGH_PAIRED__condX3
        + 2 aux_cap_helper__b0,r11,band_HIGH_PAIRED__Z3 <= 0
 c4833: - 15.8453086887446 sv_function_helpervar_b4,r1__Z3
        + sv_function_helpervar_b4,r1__condX3 <= 0
 c4834: aux_cap_helper__b5,r7,band_UNPAIRED__condX1
        + aux_cap_helper__b5,r7,band_UNPAIRED__condX2 - X__b5,r7,band_UNPAIRED
         = 0
 c4835: aux_cap_helper__b6,r2,band_LOW_PAIRED__Z2
        - aux_cap_helper__b6,r2,band_LOW_PAIRED__condX2 <= 0
 c4836: 2 aux_cap_helper__b7,r5,band_HIGH_PAIRED__Z3
        - aux_cap_helper__b7,r5,band_HIGH_PAIRED__condX3 <= 0
 c4837: - 15.8453086887446 sv_function_helpervar_b7,r10__Z3
        + sv_function_helpervar_b7,r10__condX3 <= 0
 c4838: - aux_cap_helper__b8,r3,band_UNPAIRED__condX2
        + aux_cap_helper__b8,r3,band_UNPAIRED__Z2 <= 0
 c4839: aux_cap_b5,r8,band_UNPAIRED - aux_c_b5,r8
        + aux_cap_b5,r8,band_LOW_PAIRED + aux_cap_b5,r8,band_HIGH_PAIRED  = 0
 c4840: aux_cap_helper__b2,r8,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b2,r8,band_HIGH_PAIRED__condX2 <= 0
 c4841: aux_cap_helper__b1,r9,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b1,r9,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b1,r9,band_HIGH_PAIRED__Z3  = 1
 c4842: - aux_cap_helper__b9,r12,band_LOW_PAIRED__condX2
        + aux_cap_helper__b9,r12,band_LOW_PAIRED__Z2 <= 0
 c4843: - aux_cap_helper__b5,r8,band_HIGH_PAIRED__condX1 <= 0
 c4844: - X__b3,r11,band_LOW_PAIRED
        + aux_cap_helper__b3,r11,band_LOW_PAIRED__condX2
        + aux_cap_helper__b3,r11,band_LOW_PAIRED__condX1  = 0
 c4845: aux_cap_helper__b3,r8,band_UNPAIRED__Z1
        + aux_cap_helper__b3,r8,band_UNPAIRED__Z2  = 1
 c4846: - X__b3,r6,band_HIGH_PAIRED
        + aux_cap_helper__b3,r6,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b3,r6,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b3,r6,band_HIGH_PAIRED__condX3  = 0
 c4847: 1930.09997561811 aux_sv_b4,r11 - aux_Omega_b4,r11  = 0
 c4848: - sv_function_helpervar_b7,r7__condX3
        + 8.08713064043177 sv_function_helpervar_b7,r7__Z3 <= 0
 c4849: sv_function_helpervar_b1,r4__condX1
        - 0.158453086904274 sv_function_helpervar_b1,r4__Z1 <= 0
 c4850: - 3 aux_cap_helper__b6,r1,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b6,r1,band_HIGH_PAIRED__condX3 <= 0
 c4851: aux_cap_helper__b6,r12,band_LOW_PAIRED__Z2
        - aux_cap_helper__b6,r12,band_LOW_PAIRED__condX2 <= 0
 c4852: sv_function_helpervar_b1,r5__condX3
        + sv_function_helpervar_b1,r5__condX2
        + sv_function_helpervar_b1,r5__condX1 - aux_c_b1,r5  = 0
 c4853: - aux_cap_helper__b9,r12,band_LOW_PAIRED__condX1 <= 0
 c4854: - sv_function_helpervar_b9,r7__condX2
        + 0.158453086895241 sv_function_helpervar_b9,r7__Z2 <= 0
 c4855: aux_cap_b9,r0,band_LOW_PAIRED + aux_cap_b9,r0,band_UNPAIRED
        + aux_cap_b9,r0,band_HIGH_PAIRED - aux_c_b9,r0  = 0
 c4856: - aux_cap_helper__b2,r8,band_HIGH_PAIRED__condX1 <= 0
 c4857: aux_cap_helper__b3,r6,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b3,r6,band_UNPAIRED__Z2 <= 0
 c4858: aux_cap_helper__b3,r4,band_LOW_PAIRED__condX2
        - 2 aux_cap_helper__b3,r4,band_LOW_PAIRED__Z2 <= 0
 c4859: - aux_cap_helper__b3,r12,band_LOW_PAIRED__condX2
        + aux_cap_helper__b3,r12,band_LOW_PAIRED__Z2 <= 0
 c4860: - aux_cap_helper__b6,r12,band_LOW_PAIRED__condX1 <= 0
 c4861: aux_cap_helper__b4,r11,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b4,r11,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b4,r11,band_HIGH_PAIRED__condX1
        - X__b4,r11,band_HIGH_PAIRED  = 0
 c4862: - X__b2,r8,band_UNPAIRED + aux_cap_helper__b2,r8,band_UNPAIRED__condX2
        + aux_cap_helper__b2,r8,band_UNPAIRED__condX1  = 0
 c4863: aux_cap_helper__b2,r12,band_UNPAIRED__condX1
        + aux_cap_helper__b2,r12,band_UNPAIRED__condX2
        - X__b2,r12,band_UNPAIRED  = 0
 c4864: aux_cap_helper__b0,r8,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b0,r8,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b0,r8,band_HIGH_PAIRED__Z1  = 1
 c4865: - 0.8112667416 aux_cap_helper__b8,r4,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b8,r4,band_HIGH_PAIRED__Z3
        - aux_cap_b8,r4,band_HIGH_PAIRED
        + 2.0281668543 aux_cap_helper__b8,r4,band_HIGH_PAIRED__condX1
        + 2.8394335959 aux_cap_helper__b8,r4,band_HIGH_PAIRED__condX2
        + 1.8253501688 aux_cap_helper__b8,r4,band_HIGH_PAIRED__condX3  = 0
 c4866: - aux_cap_helper__b3,r12,band_LOW_PAIRED__condX1 <= 0
 c4867: - aux_c_b6,r11 + sv_function_helpervar_b6,r11__condX3
        + sv_function_helpervar_b6,r11__condX1
        + sv_function_helpervar_b6,r11__condX2  = 0
 c4868: aux_cap_b0,r1,band_LOW_PAIRED + aux_cap_b0,r1,band_UNPAIRED
        - aux_c_b0,r1 + aux_cap_b0,r1,band_HIGH_PAIRED  = 0
 c4869: sv_function_helpervar_b4,r13__condX1
        - 0.158453086947509 sv_function_helpervar_b4,r13__Z1 <= 0
 c4870: - 2 aux_cap_helper__b8,r12,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b8,r12,band_HIGH_PAIRED__condX2 <= 0
 c4871: - aux_cap_helper__b7,r11,band_LOW_PAIRED__condX1 <= 0
 c4872: aux_cap_helper__b2,r8,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b2,r8,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b2,r8,band_HIGH_PAIRED__Z2  = 1
 c4873: - aux_cap_helper__b7,r13,band_UNPAIRED__condX2
        + aux_cap_helper__b7,r13,band_UNPAIRED__Z2 <= 0
 c4874: - sv_function_helpervar_b3,r9__condX3
        + 5.42311341710275 sv_function_helpervar_b3,r9__Z3 <= 0
 c4875: aux_cap_helper__b6,r1,band_LOW_PAIRED__Z1
        + aux_cap_helper__b6,r1,band_LOW_PAIRED__Z2  = 1
 c4876: aux_cap_helper__b1,r6,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b1,r6,band_HIGH_PAIRED__Z1 <= 0
 c4877: sv_function_helpervar_b9,r7__Z3 + sv_function_helpervar_b9,r7__Z1
        + sv_function_helpervar_b9,r7__Z2  = 1
 c4878: aux_cap_helper__b6,r8,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b6,r8,band_UNPAIRED__Z2 <= 0
 c4879: aux_cap_helper__b2,r3,band_LOW_PAIRED__Z2
        + aux_cap_helper__b2,r3,band_LOW_PAIRED__Z1  = 1
 c4880: 0.158453086941154 sv_function_helpervar_b6,r8__Z2
        - sv_function_helpervar_b6,r8__condX2 <= 0
 c4881: - aux_cap_helper__b5,r10,band_LOW_PAIRED__condX2
        + aux_cap_helper__b5,r10,band_LOW_PAIRED__Z2 <= 0
 c4882: aux_cap_helper__b8,r1,band_LOW_PAIRED__condX1
        - aux_cap_helper__b8,r1,band_LOW_PAIRED__Z1 <= 0
 c4883: - 2 aux_cap_helper__b5,r12,band_UNPAIRED__Z2
        + aux_cap_helper__b5,r12,band_UNPAIRED__condX2 <= 0
 c4884: - 2 aux_cap_helper__b4,r6,band_LOW_PAIRED__Z2
        + aux_cap_helper__b4,r6,band_LOW_PAIRED__condX2 <= 0
 c4885: - aux_cap_b6,r12,band_UNPAIRED
        - 0.253841888 aux_cap_helper__b6,r12,band_UNPAIRED__Z2
        + 0.888446608 aux_cap_helper__b6,r12,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b6,r12,band_UNPAIRED__condX1  = 0
 c4886: aux_cap_helper__b2,r13,band_LOW_PAIRED__Z2
        + aux_cap_helper__b2,r13,band_LOW_PAIRED__Z1  = 1
 c4887: aux_cap_helper__b1,r13,band_HIGH_PAIRED__condX1
        - aux_cap_helper__b1,r13,band_HIGH_PAIRED__Z1 <= 0
 c4888: - aux_c_b8,r5 + sv_function_helpervar_b8,r5__condX1
        + sv_function_helpervar_b8,r5__condX2
        + sv_function_helpervar_b8,r5__condX3  = 0
 c4889: X__b5,r1,band_LOW_PAIRED + X__b1,r1,band_LOW_PAIRED
        + X__b7,r1,band_LOW_PAIRED + X__b9,r1,band_LOW_PAIRED
        + X__b4,r1,band_LOW_PAIRED + X__b2,r1,band_LOW_PAIRED
        + X__b6,r1,band_LOW_PAIRED + X__b8,r1,band_LOW_PAIRED
        + X__b0,r1,band_LOW_PAIRED + X__b3,r1,band_LOW_PAIRED <= 2
 c4890: - aux_cap_helper__b5,r10,band_LOW_PAIRED__condX1 <= 0
 c4891: - aux_cap_helper__b7,r0,band_HIGH_PAIRED__condX1 <= 0
 c4892: - X__b1,r3,band_LOW_PAIRED
        + aux_cap_helper__b1,r3,band_LOW_PAIRED__condX2
        + aux_cap_helper__b1,r3,band_LOW_PAIRED__condX1  = 0
 c4893: - sv_function_helpervar_b5,r5__condX3
        + 6.16584813276107 sv_function_helpervar_b5,r5__Z3 <= 0
 c4894: 0.0102040816326531 X__b8,r8,band_HIGH_PAIRED
        + 0.0102040816326531 X__b8,r8,band_LOW_PAIRED
        - GlobalBidder_W_ir__b8,r8 + 0.0102040816326531 X__b8,r8,band_UNPAIRED
        <= 0
 c4895: - aux_cap_helper__b2,r10,band_LOW_PAIRED__condX1 <= 0
 c4896: aux_cap_helper__b1,r0,band_HIGH_PAIRED__Z2
        - aux_cap_helper__b1,r0,band_HIGH_PAIRED__condX2 <= 0
 c4897: sv_function_helpervar_b6,r9__condX1
        - 0.158453086948978 sv_function_helpervar_b6,r9__Z1 <= 0
 c4898: 0.0102040816326531 X__b7,r13,band_HIGH_PAIRED
        + 0.0102040816326531 X__b7,r13,band_UNPAIRED
        + 0.0102040816326531 X__b7,r13,band_LOW_PAIRED
        - GlobalBidder_W_ir__b7,r13 <= 0
 c4899: 0.15845308689803 sv_function_helpervar_b1,r0__Z2
        - sv_function_helpervar_b1,r0__condX2 <= 0
 c4900: aux_cap_helper__b4,r8,band_HIGH_PAIRED__condX2
        + aux_cap_helper__b4,r8,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b4,r8,band_HIGH_PAIRED__condX1
        - X__b4,r8,band_HIGH_PAIRED  = 0
 c4901: - 2 aux_cap_helper__b4,r9,band_UNPAIRED__Z2
        + aux_cap_helper__b4,r9,band_UNPAIRED__condX2 <= 0
 c4902: aux_cap_helper__b6,r11,band_UNPAIRED__condX2
        + aux_cap_helper__b6,r11,band_UNPAIRED__condX1
        - X__b6,r11,band_UNPAIRED  = 0
 c4903: - 1.2715511236 aux_cap_helper__b8,r9,band_LOW_PAIRED__Z2
        + 3.1788778092 aux_cap_helper__b8,r9,band_LOW_PAIRED__condX1
        + 4.4504289328 aux_cap_helper__b8,r9,band_LOW_PAIRED__condX2
        - aux_cap_b8,r9,band_LOW_PAIRED  = 0
 c4904: - aux_cap_helper__b1,r0,band_HIGH_PAIRED__condX1 <= 0
 c4905: - X__b6,r9,band_LOW_PAIRED
        + aux_cap_helper__b6,r9,band_LOW_PAIRED__condX1
        + aux_cap_helper__b6,r9,band_LOW_PAIRED__condX2  = 0
 c4906: 2.8394335959 aux_cap_helper__b1,r13,band_HIGH_PAIRED__condX2
        + 2.0281668543 aux_cap_helper__b1,r13,band_HIGH_PAIRED__condX1
        + 1.8253501688 aux_cap_helper__b1,r13,band_HIGH_PAIRED__condX3
        - aux_cap_b1,r13,band_HIGH_PAIRED
        - 0.8112667416 aux_cap_helper__b1,r13,band_HIGH_PAIRED__Z2
        + 1.21690011250031 aux_cap_helper__b1,r13,band_HIGH_PAIRED__Z3  = 0
 c4907: - aux_cap_helper__b2,r12,band_UNPAIRED__condX2
        + aux_cap_helper__b2,r12,band_UNPAIRED__Z2 <= 0
 c4908: sv_function_helpervar_b3,r11__Z2 + sv_function_helpervar_b3,r11__Z3
        + sv_function_helpervar_b3,r11__Z1  = 1
 c4909: - 5.91034954743343 sv_function_helpervar_b4,r0__Z2
        + sv_function_helpervar_b4,r0__condX2 <= 0
 c4910: - aux_cap_b2,r12,band_UNPAIRED
        + 0.63460472 aux_cap_helper__b2,r12,band_UNPAIRED__condX1
        + 0.888446608 aux_cap_helper__b2,r12,band_UNPAIRED__condX2
        - 0.253841888 aux_cap_helper__b2,r12,band_UNPAIRED__Z2  = 0
 c4911: - aux_cap_helper__b1,r11,band_LOW_PAIRED__condX2
        + aux_cap_helper__b1,r11,band_LOW_PAIRED__Z2 <= 0
 c4912: 17.5172023551 sv_function_helpervar_b3,r9__condX3
        + 1152.1877366228 sv_function_helpervar_b3,r9__condX1
        + 59.080976401 sv_function_helpervar_b3,r9__condX2
        + 173.206140469816 sv_function_helpervar_b3,r9__Z2
        + 398.611201163552 sv_function_helpervar_b3,r9__Z3 - aux_sv_b3,r9  = 0
 c4913: aux_cap_helper__b6,r2,band_UNPAIRED__Z2
        + aux_cap_helper__b6,r2,band_UNPAIRED__Z1  = 1
 c4914: - aux_cap_helper__b4,r11,band_LOW_PAIRED__condX1 <= 0
 c4915: aux_cap_b6,r6,band_HIGH_PAIRED + aux_cap_b6,r6,band_UNPAIRED
        + aux_cap_b6,r6,band_LOW_PAIRED - aux_c_b6,r6  = 0
 c4916: aux_cap_helper__b7,r10,band_LOW_PAIRED__Z1
        + aux_cap_helper__b7,r10,band_LOW_PAIRED__Z2  = 1
 c4917: aux_cap_helper__b8,r8,band_HIGH_PAIRED__Z1
        + aux_cap_helper__b8,r8,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b8,r8,band_HIGH_PAIRED__Z2  = 1
 c4918: - aux_cap_helper__b7,r13,band_UNPAIRED__condX1 <= 0
 c4919: aux_cap_helper__b8,r13,band_HIGH_PAIRED__condX1
        + aux_cap_helper__b8,r13,band_HIGH_PAIRED__condX3
        + aux_cap_helper__b8,r13,band_HIGH_PAIRED__condX2
        - X__b8,r13,band_HIGH_PAIRED  = 0
 c4920: sv_function_helpervar_b9,r4__condX2
        - 7.52273391768105 sv_function_helpervar_b9,r4__Z2 <= 0
 c4921: - aux_c_b0,r13 + aux_cap_b0,r13,band_UNPAIRED
        + aux_cap_b0,r13,band_LOW_PAIRED + aux_cap_b0,r13,band_HIGH_PAIRED  = 0
 c4922: - aux_cap_b6,r13,band_LOW_PAIRED
        - 1.2715511236 aux_cap_helper__b6,r13,band_LOW_PAIRED__Z2
        + 3.1788778092 aux_cap_helper__b6,r13,band_LOW_PAIRED__condX1
        + 4.4504289328 aux_cap_helper__b6,r13,band_LOW_PAIRED__condX2  = 0
 c4923: - aux_cap_b6,r1,band_LOW_PAIRED
        - 1.2715511236 aux_cap_helper__b6,r1,band_LOW_PAIRED__Z2
        + 4.4504289328 aux_cap_helper__b6,r1,band_LOW_PAIRED__condX2
        + 3.1788778092 aux_cap_helper__b6,r1,band_LOW_PAIRED__condX1  = 0
 c4924: 23.7933667772 sv_function_helpervar_b7,r6__condX3
        + 46.2788338373 sv_function_helpervar_b7,r6__condX2
        + 1255.6694400892 sv_function_helpervar_b7,r6__condX1
        + 191.631674826423 sv_function_helpervar_b7,r6__Z2
        + 359.893050916263 sv_function_helpervar_b7,r6__Z3 - aux_sv_b7,r6  = 0
 c4925: - aux_cap_b0,r12,band_UNPAIRED
        - 0.253841888 aux_cap_helper__b0,r12,band_UNPAIRED__Z2
        + 0.63460472 aux_cap_helper__b0,r12,band_UNPAIRED__condX1
        + 0.888446608 aux_cap_helper__b0,r12,band_UNPAIRED__condX2  = 0
 c4926: - 2 aux_cap_helper__b0,r11,band_UNPAIRED__Z2
        + aux_cap_helper__b0,r11,band_UNPAIRED__condX2 <= 0
 c4927: 0.888446608 aux_cap_helper__b4,r6,band_UNPAIRED__condX2
        + 0.63460472 aux_cap_helper__b4,r6,band_UNPAIRED__condX1
        - aux_cap_b4,r6,band_UNPAIRED
        - 0.253841888 aux_cap_helper__b4,r6,band_UNPAIRED__Z2  = 0
 c4928: - 2 aux_cap_helper__b5,r8,band_LOW_PAIRED__Z2
        + aux_cap_helper__b5,r8,band_LOW_PAIRED__condX2 <= 0
 c4929: - aux_cap_b6,r7,band_HIGH_PAIRED
        + 1.21690011250031 aux_cap_helper__b6,r7,band_HIGH_PAIRED__Z3
        - 0.8112667416 aux_cap_helper__b6,r7,band_HIGH_PAIRED__Z2
        + 2.0281668543 aux_cap_helper__b6,r7,band_HIGH_PAIRED__condX1
        + 2.8394335959 aux_cap_helper__b6,r7,band_HIGH_PAIRED__condX2
        + 1.8253501688 aux_cap_helper__b6,r7,band_HIGH_PAIRED__condX3  = 0
 c4930: - aux_c_b7,r4 + aux_cap_b7,r4,band_UNPAIRED
        + aux_cap_b7,r4,band_LOW_PAIRED + aux_cap_b7,r4,band_HIGH_PAIRED  = 0
 c4931: aux_cap_helper__b8,r9,band_UNPAIRED__condX2
        + aux_cap_helper__b8,r9,band_UNPAIRED__condX1 - X__b8,r9,band_UNPAIRED
         = 0
 c4932: aux_cap_helper__b8,r9,band_LOW_PAIRED__Z2
        - aux_cap_helper__b8,r9,band_LOW_PAIRED__condX2 <= 0
 c4933: aux_cap_helper__b0,r8,band_LOW_PAIRED__condX1
        - aux_cap_helper__b0,r8,band_LOW_PAIRED__Z1 <= 0
 c4934: aux_cap_helper__b2,r7,band_UNPAIRED__Z1
        + aux_cap_helper__b2,r7,band_UNPAIRED__Z2  = 1
 c4935: - sv_function_helpervar_b8,r2__condX2
        + 0.158453086941469 sv_function_helpervar_b8,r2__Z2 <= 0
 c4936: sv_function_helpervar_b5,r8__Z1 + sv_function_helpervar_b5,r8__Z2
        + sv_function_helpervar_b5,r8__Z3  = 1
 c4937: - 2 aux_cap_helper__b6,r1,band_UNPAIRED__Z2
        + aux_cap_helper__b6,r1,band_UNPAIRED__condX2 <= 0
 c4938: - 15.8453086887446 sv_function_helpervar_b9,r1__Z3
        + sv_function_helpervar_b9,r1__condX3 <= 0
 c4939: - 1.2715511236 aux_cap_helper__b1,r6,band_LOW_PAIRED__Z2
        - aux_cap_b1,r6,band_LOW_PAIRED
        + 3.1788778092 aux_cap_helper__b1,r6,band_LOW_PAIRED__condX1
        + 4.4504289328 aux_cap_helper__b1,r6,band_LOW_PAIRED__condX2  = 0
 c4940: - aux_cap_helper__b7,r11,band_LOW_PAIRED__condX2
        + aux_cap_helper__b7,r11,band_LOW_PAIRED__Z2 <= 0
 c4941: aux_cap_helper__b5,r5,band_UNPAIRED__condX2
        - 2 aux_cap_helper__b5,r5,band_UNPAIRED__Z2 <= 0
 c4942: aux_cap_helper__b6,r8,band_HIGH_PAIRED__Z2
        + aux_cap_helper__b6,r8,band_HIGH_PAIRED__Z3
        + aux_cap_helper__b6,r8,band_HIGH_PAIRED__Z1  = 1
 c4943: - X__b8,r7,band_LOW_PAIRED
        + aux_cap_helper__b8,r7,band_LOW_PAIRED__condX2
        + aux_cap_helper__b8,r7,band_LOW_PAIRED__condX1  = 0
 c4944: - aux_cap_helper__b8,r9,band_LOW_PAIRED__condX1 <= 0
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
 0 <= aux_cap_helper__b7,r6,band_HIGH_PAIRED__condX2 <= 3
 0 <= aux_cap_helper__b7,r4,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b7,r6,band_HIGH_PAIRED__condX1 <= 3
 0 <= aux_cap_helper__b7,r4,band_LOW_PAIRED__condX1 <= 2
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
 0 <= X__b6,r9,band_HIGH_PAIRED <= 3
 0 <= X__b8,r12,band_UNPAIRED <= 2
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
 0 <= aux_cap_helper__b8,r1,band_UNPAIRED__condX1 <= 2
 0 <= aux_cap_helper__b8,r2,band_LOW_PAIRED__Z2 <= 1
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
 0 <= aux_cap_helper__b8,r2,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b8,r1,band_LOW_PAIRED__condX2 <= 2
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
 0 <= aux_cap_helper__b6,r13,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b6,r10,band_UNPAIRED__Z2 <= 1
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
 0 <= X__b6,r11,band_UNPAIRED <= 2
 0 <= X__b8,r4,band_LOW_PAIRED <= 2
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
 0 <= X__b5,r2,band_LOW_PAIRED <= 2
 0 <= X__b7,r5,band_LOW_PAIRED <= 2
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
 0 <= X__b3,r7,band_UNPAIRED <= 2
 0 <= X__b1,r11,band_HIGH_PAIRED <= 3
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
 0 <= aux_cap_helper__b0,r3,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b0,r0,band_HIGH_PAIRED__condX1 <= 3
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
 0 <= X__b5,r3,band_LOW_PAIRED <= 2
 0 <= X__b1,r13,band_LOW_PAIRED <= 2
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
 0 <= aux_cap_helper__b6,r2,band_HIGH_PAIRED__Z2 <= 1
 0 <= aux_cap_helper__b6,r3,band_UNPAIRED__condX2 <= 2
 0 <= aux_cap_helper__b6,r2,band_HIGH_PAIRED__Z3 <= 1
 0 <= aux_cap_helper__b6,r3,band_UNPAIRED__condX1 <= 2
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
 0 <= aux_cap_helper__b6,r10,band_LOW_PAIRED__condX2 <= 2
 0 <= aux_cap_helper__b6,r2,band_UNPAIRED__condX2 <= 2
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
 X__b6,r9,band_HIGH_PAIRED  X__b8,r12,band_UNPAIRED  X__b8,r6,band_HIGH_PAIRED 
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
 X__b9,r13,band_LOW_PAIRED  X__b6,r3,band_UNPAIRED  X__b6,r11,band_UNPAIRED 
 X__b8,r4,band_LOW_PAIRED  X__b7,r9,band_UNPAIRED  X__b2,r13,band_LOW_PAIRED 
 X__b5,r0,band_HIGH_PAIRED  X__b9,r10,band_HIGH_PAIRED 
 X__b2,r1,band_HIGH_PAIRED  X__b1,r2,band_LOW_PAIRED  X__b5,r2,band_LOW_PAIRED 
 X__b7,r5,band_LOW_PAIRED  X__b7,r9,band_HIGH_PAIRED 
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
 X__b3,r7,band_UNPAIRED  X__b1,r11,band_HIGH_PAIRED  X__b0,r8,band_UNPAIRED 
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
 X__b5,r3,band_LOW_PAIRED  X__b1,r13,band_LOW_PAIRED  X__b9,r6,band_UNPAIRED 
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
