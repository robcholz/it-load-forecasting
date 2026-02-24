## History Length (Context)
| Model | History Length |
| --- | --- |
| chronos | 672 |
| timecma | 96 |
| timemoe | 672 |
| timerxl | 672 |
| timesfm | 672 |

## Horizon 96
| Dataset | chronos (ctx=672) MSE | chronos (ctx=672) MAE | timecma (ctx=96) MSE | timecma (ctx=96) MAE | timemoe (ctx=672) MSE | timemoe (ctx=672) MAE | timerxl (ctx=672) MSE | timerxl (ctx=672) MAE | timesfm (ctx=672) MSE | timesfm (ctx=672) MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh1 | 0.424136 | **0.382852** | 0.392219 | 0.413046 | 0.397899 | 0.390283 | **0.375721** | 0.387947 | 0.395447 | 0.384662 |
| ETTh2 | 0.306949 | **0.318795** | 0.329715 | 0.365193 | **0.278061** | 0.346459 | 0.287494 | 0.341790 | 0.291163 | 0.332555 |
| ETTm1 | 0.352476 | 0.324191 | 0.334431 | 0.372504 | **0.279073** | 0.338189 | 0.364240 | 0.368856 | 0.325437 | **0.322430** |
| ETTm2 | 0.176046 | **0.232801** | 0.182537 | 0.263719 | **0.172586** | 0.265625 | 0.194250 | 0.274029 | 0.179013 | 0.240495 |

Best by dataset:
- ETTh1: MSE -> timerxl; MAE -> chronos
- ETTh2: MSE -> timemoe; MAE -> chronos
- ETTm1: MSE -> timemoe; MAE -> timesfm
- ETTm2: MSE -> timemoe; MAE -> chronos

## Horizon 192
| Dataset | chronos (ctx=672) MSE | chronos (ctx=672) MAE | timecma (ctx=96) MSE | timecma (ctx=96) MAE | timemoe (ctx=672) MSE | timemoe (ctx=672) MAE | timerxl (ctx=672) MSE | timerxl (ctx=672) MAE | timesfm (ctx=672) MSE | timesfm (ctx=672) MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh1 | 0.436556 | **0.402632** | 0.432702 | 0.435447 | 0.431699 | 0.426459 | 0.417113 | 0.411213 | **0.402029** | 0.404010 |
| ETTh2 | **0.366603** | **0.377621** | 0.408873 | 0.413214 | 0.410206 | 0.424409 | 0.385008 | 0.398558 | 0.398200 | 0.391575 |
| ETTm1 | 0.409714 | 0.365028 | **0.383055** | 0.396113 | 0.412130 | 0.419952 | 0.424433 | 0.399579 | 0.385151 | **0.360208** |
| ETTm2 | **0.246043** | **0.279264** | 0.256286 | 0.315087 | 0.280088 | 0.347602 | 0.246435 | 0.313366 | 0.246193 | 0.286992 |

Best by dataset:
- ETTh1: MSE -> timesfm; MAE -> chronos
- ETTh2: MSE -> chronos; MAE -> chronos
- ETTm1: MSE -> timecma; MAE -> timesfm
- ETTm2: MSE -> chronos; MAE -> chronos

## Horizon 336
| Dataset | chronos (ctx=672) MSE | chronos (ctx=672) MAE | timecma (ctx=96) MSE | timecma (ctx=96) MAE | timemoe (ctx=672) MSE | timemoe (ctx=672) MAE | timerxl (ctx=672) MSE | timerxl (ctx=672) MAE | timesfm (ctx=672) MSE | timesfm (ctx=672) MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh1 | 0.537068 | 0.438550 | 0.468725 | 0.453753 | **0.467380** | 0.448564 | 0.469175 | **0.431455** | 0.514918 | 0.438834 |
| ETTh2 | **0.360453** | **0.378403** | 0.440737 | 0.444059 | 0.564638 | 0.486154 | 0.395474 | 0.404872 | 0.394452 | 0.397815 |
| ETTm1 | 0.437854 | 0.382424 | 0.423274 | 0.417659 | 0.454966 | 0.465744 | 0.455135 | 0.425926 | **0.410723** | **0.381272** |
| ETTm2 | 0.359551 | 0.349045 | **0.311603** | **0.348698** | 0.486863 | 0.463485 | 0.351454 | 0.364681 | 0.363218 | 0.355678 |

Best by dataset:
- ETTh1: MSE -> timemoe; MAE -> timerxl
- ETTh2: MSE -> chronos; MAE -> chronos
- ETTm1: MSE -> timesfm; MAE -> timesfm
- ETTm2: MSE -> timecma; MAE -> timecma

## Horizon 720
| Dataset | chronos (ctx=672) MSE | chronos (ctx=672) MAE | timecma (ctx=96) MSE | timecma (ctx=96) MAE | timemoe (ctx=672) MSE | timemoe (ctx=672) MAE | timerxl (ctx=672) MSE | timerxl (ctx=672) MAE | timesfm (ctx=672) MSE | timesfm (ctx=672) MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh1 | 0.478514 | 0.445628 | 0.468056 | 0.469614 | 0.591685 | 0.566049 | 0.475226 | 0.470643 | **0.453510** | **0.438085** |
| ETTh2 | 0.543286 | 0.461140 | 0.448520 | 0.457707 | 0.986505 | 0.639948 | 0.424015 | 0.430248 | **0.405695** | **0.426932** |
| ETTm1 | 0.568351 | 0.448362 | **0.472160** | 0.449819 | 0.723019 | 0.603138 | 0.584168 | 0.490538 | 0.510582 | **0.430353** |
| ETTm2 | 0.431067 | **0.395413** | 0.425834 | 0.411200 | 0.914376 | 0.633476 | **0.418040** | 0.411555 | 0.434591 | 0.399317 |

Best by dataset:
- ETTh1: MSE -> timesfm; MAE -> timesfm
- ETTh2: MSE -> timesfm; MAE -> timesfm
- ETTm1: MSE -> timecma; MAE -> timesfm
- ETTm2: MSE -> timerxl; MAE -> chronos
