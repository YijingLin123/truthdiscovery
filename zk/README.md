  - python export_input.py
  - python verify.py
  - python truthfinder_agent_fixed.py
  - circom truthfinder_top.circom --r1cs --wasm --prime bn128
  - node truthfinder_top_js/generate_witness.js truthfinder_top_js/
    truthfinder_top.wasm ../truthfinder_input.json ../witness.wtns


https://docs.circom.io/getting-started/installation/#installing-snarkjs

根据文档命令安装circom和snarkjs

1. 生成输入

(acon) ➜  zk git:(master) ✗ python export_input.py
Wrote Circom input to /Users/yijing/Desktop/区块链语义预言机/TruthFinder/zk/truthfinder_input.json

2. 编译电路（确认 circom --version 为 2.x）
circom truthfinder_top.circom --r1cs --wasm --prime bn128

3. 生成并检查 witness
node truthfinder_top_js/generate_witness.js truthfinder_top_js/truthfinder_top.wasm ../truthfinder_input.json ../witness.wtns
4. 验证
snarkjs wtns check truthfinder_top.r1cs ../witness.wtns

