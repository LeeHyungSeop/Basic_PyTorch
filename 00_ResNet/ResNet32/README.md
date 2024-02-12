1. `My_ResNet32_exp1/` : 
   - option (B) : projection shortcuts are used for increasing dimension
   - Top-1 acc : 92.12%
2. `My_ResNet32_exp2/` : 
   - 논문에서 사용한  방법
   - option (A) : projection shortcuts are used for increasing dimension
   - Top-1 acc : 92.24% (Paper : 92.49%)
3. `My_ResNet32_exp3/` : 
   - exp2에서 architecutre 구조만 변경
     - 변경 전 : BN -> ReLU
     - 변경 후 : ReLU -> BN
     - Top-1 acc : 91.66% (Paper : 92.49%)