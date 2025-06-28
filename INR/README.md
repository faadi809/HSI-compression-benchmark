This code implements the method proposed by:
Rezasoltani, S., & Qureshi, F. Z. in
"Hyperspectral image compression using implicit neural representations" at 20th Conference on Robots and Vision (CRV) 2023, (pp. 248-255). IEEE.

Change the path to PaviaU.mat.
You can change the values of hidden_dim and hidden_layers for different bitrates.
model = SIREN(in_dim=2, hidden_dim=512, out_dim=C, hidden_layers=4)
