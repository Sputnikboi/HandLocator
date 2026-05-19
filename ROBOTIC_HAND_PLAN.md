# Robotic Hand Build Plan

## Overview
5-finger, 10-DOF robotic hand using Feetech STS3032 serial bus servos, 3D printed on a Prusa Core One, controlled via HandLocator's vision pipeline.

---

## Bill of Materials

### Servos & Electronics
| Item | Qty | Est. Cost | Notes |
|---|---|---|---|
| Feetech STS3032 servo | 10 | ~$220 | 2 per finger (curl + spread). Awaiting Alibaba quote. |
| Feetech FE-URT-1 debug board | 1 | ~$8 | USB-to-TTL signal converter only (servo power is separate) |
| 6V 5A+ DC power supply | 1 | ~$15 | Powers servo daisy chain directly. STS3032 runs at 6V. |
| USB cable (Mini USB to PC) | 1 | ~$5 | Connects URT-1 to PC (shows up as serial port) |
| **Subtotal** | | **~$248** | |

### Mechanical / 3D Printing
| Item | Qty | Est. Cost | Notes |
|---|---|---|---|
| PLA filament | ~200g | ~$5 | Palm, finger bones, brackets. Print on Prusa Core One. |
| TPU filament | ~50g | ~$5 | Fingertip pads for grip (optional, Prusa can do this) |
| M2 screw assortment (4-10mm) | 1 pack | ~$8 | Servo mounting, joint pins |
| M2 brass heat-set inserts | 1 pack | ~$7 | Threaded mounting points in PLA |
| M3 screw assortment | 1 pack | ~$8 | Palm assembly, mounting plate |
| Bearing pins or shoulder screws (2mm shaft) | ~20 | ~$5 | Finger joint pivots |
| Rubber/silicone fingertip pads | 5 | ~$3 | Grip friction (or print in TPU) |
| **Subtotal** | | **~$41** | |

### Tools (you probably have most of these)
| Item | Notes |
|---|---|
| Soldering iron | For brass inserts + optional wiring |
| M2 / M2.5 / M3 hex wrenches | Servo mounting |
| Calipers | Measuring servo dimensions for CAD |
| Multimeter | Checking power, debugging |

### **Total Estimated Cost: ~$290**

---

## Design Decisions to Make

### 1. Finger Mechanism
**Recommended: Underactuated linkage with 2 DOF per finger**
- Motor 1: Curl (flexion/extension) — drives MCP, PIP, DIP joints through a linkage
- Motor 2: Spread (abduction/adduction) — side-to-side at MCP
- The Amazing Hand's differential approach is worth studying: both motors work together for stronger curl when spread isn't needed

### 2. Servo Placement
Two options:
- **In-palm** — all 10 servos housed in the palm body. Compact but bulky palm.
- **In-finger base** — curl servo at MCP joint, spread servo in palm. More anthropomorphic but wider fingers.

STS3032 is 32×12×27.5mm — small enough for either approach. Measure and prototype one finger first.

### 3. Thumb Design
The thumb needs special attention:
- Needs opposition (ability to face other fingers)
- Consider 3 DOF for the thumb (curl, spread, rotation) = 1 extra servo
- Or: use 2 servos + a fixed opposition angle built into the palm geometry

### 4. Mounting
- Start with a flat base / stand for tabletop use
- Design the wrist mount as a separate plate — easy to swap for a robot arm adapter later
- LEAP V1's top plate design (freely available STL/STP) is a good reference

---

## Software Pipeline

### What exists now (HandLocator)
- [x] YOLOv8n hand detector (96.9% mAP50)
- [x] MediaPipe landmark model (21 points, screen + world space)
- [x] Multi-hand tracking with smoothed ROI
- [x] GLFW 3D viewer with world-space landmarks
- [x] Presence/handedness detection

### What needs to be built

#### Stage 1: Serial Servo Control
- [ ] `servo_controller.hpp/cpp` — Feetech STS protocol over serial
  - Initialize bus, set IDs, read/write positions
  - Feetech SDK: https://github.com/nicholaswilson/SCServo (C++) or use the Python SDK first for testing
  - LeRobot's Feetech support is also a reference: pip install lerobot
- [ ] Calibration routine — find min/max positions for each servo
- [ ] Test script — move each finger independently

#### Stage 2: Landmark → Joint Angle Retargeting
- [ ] `retarget.hpp/cpp` — convert 21 MediaPipe landmarks to 10 servo angles
  - Compute joint angles from landmark positions using vectors/dot products:
    - Finger curl angle = angle between (MCP→PIP) and (PIP→DIP) vectors
    - Finger spread angle = angle between finger direction and palm normal
  - Map human joint angles → servo angle range (calibrated min/max)
  - This is simpler than full IK — direct geometric mapping from landmarks
- [ ] Per-finger gain/offset tuning (human hand ≠ robot hand proportions)

#### Stage 3: Live Teleoperation
- [ ] Real-time loop: camera → landmarks → retarget → servo commands
- [ ] Smooth servo commands (low-pass filter to avoid jitter)
- [ ] Add latency compensation if needed
- [ ] Safety limits — max speed, max torque, emergency stop on 'q'

#### Stage 4: Data Recording
- [ ] Record timestamped streams: landmarks + servo angles + camera frames
- [ ] Save as CSV or HDF5 for training
- [ ] Add workspace camera (second camera looking at the table/objects)
- [ ] Label task segments (grasp, lift, place, release)

#### Stage 5: Policy Learning (later)
- [ ] Behavior cloning from recorded demonstrations
- [ ] Simulation environment (Isaac Gym / MuJoCo with URDF of your hand)
- [ ] Sim-to-real transfer

---

## Suggested Build Order

1. **Order servos** — waiting on quote (Alibaba / sales038@feetechrc.com)
2. **While waiting: CAD one finger** — model the STS3032 dimensions, design a 2-DOF finger with underactuated curl linkage. Print and test manually.
3. **Servos arrive: test one servo** — wire up driver board, run Feetech debug software or Python SDK, verify position control and feedback.
4. **Build one finger with servo** — validate mechanism, test range of motion.
5. **Design full palm** — arrange 10 servos, design palm body, print.
6. **Assemble full hand** — wire servos on daisy chain, calibrate.
7. **Write servo controller in C++** — integrate into HandLocator.
8. **Retargeting** — map landmarks to servo angles, tune gains.
9. **Live teleoperation** — control the hand with your hand in real-time.
10. **Record demonstrations** — pick up dice, cards, etc.

---

## Reference Links
- Feetech STS3032 specs: 4.5 kg·cm, 32×12×27.5mm, 12-bit magnetic encoder, TTL bus
- Feetech sales: sales038@feetechrc.com
- LEAP V1 CAD (free, reference): https://v1.leaphand.com/leap_cad
- LEAP V1 API (Python, reference): https://github.com/leap-hand/LEAP_Hand_API
- Amazing Hand (reference for differential mechanism): https://github.com/pollen-robotics/AmazingHand
- Feetech Python SDK: https://pypi.org/project/vassar-feetech-servo-sdk/
- LeRobot Feetech support: https://github.com/huggingface/lerobot
- SCServo C++ library: https://gitee.com/ftservo/SCServoSDK

---

## Budget Summary
| Category | Cost |
|---|---|
| Servos & electronics | ~$250 |
| Mechanical & printing | ~$41 |
| **Total** | **~$290** |
| **Remaining from $500 budget** | **~$210 (arm fund)** |
