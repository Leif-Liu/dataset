**Summary of Test Cases for Autonomous Driving System Verification**

The following test cases validate alerts, system responses, and user interface interactions for autonomous driving features, focusing on Power Mode (PM) states, CAN signals, and MFC simulator actions:

---

### **1. Alert Activation/Deactivation by Power Mode**
- **TC_Cluster_Alert_2148_Basic_RUN2**:  
  - **Condition**: PM = RUN.  
  - **Action**: Set CAN signals for "Autonomous Driving Lane Change Complete Left" alert.  
  - **Verify**: Alert remains active when PM changes to PROPULSION and deactivates within 3 seconds.  

- **TC_Cluster_Alert_2148_Basic_OFF**:  
  - **Condition**: PM = OFF.  
  - **Action**: Set CAN signals for "Lane Change Complete Left" alert.  
  - **Verify**: Alert deactivates when PM switches to OFF.  

- **TC_Cluster_Alert_536_PM_PROPULSION**:  
  - **Condition**: PM = PROPULSION.  
  - **Action**: Test Lane Change Alert System Off activation/deactivation.  
  - **Verify**: Alert behavior aligns with PM transitions (ACC to RUN).  

---

### **2. MFC Simulator Interactions**
- **TC_MFC_Autonomous_Driving_Dev_0008**:  
  - **Action**: Nudge MFC knob downward in carpool lanes.  
  - **Verify**: No unintended actions; screen navigation follows design sketches.  

- **TC_Generic_Autonomous_Driving_Dev_0112**:  
  - **Action**: Nudge MFC knob upward during lane merge.  
  - **Verify**: Focus shifts upward per theme guidelines.  

- **TC_MFC_Settings_Dev_0248**:  
  - **Action**: Rotate MFC knob anti-clockwise in Open Lane Acceleration.  
  - **Verify**: Alerts cycle (e.g., "Auto Lane Change Changing Lanes" and "Brakes Overheated") every 12 seconds.  

---

### **3. CAN Signal Validations**
- **TC_Cluster_Alert_2135_OtherDefault**:  
  - **Action**: Set PM = Propulsion; disable default display of "Lane Change Unable Right" alert.  
  - **Verify**: Alert does not appear by default.  

- **TC_Virtual_Controls_DEV_1279**:  
  - **Action**: Send CAN signals for Lane Centering Assistance in Drive/Park mode.  
  - **Verify**: Navigation to Lane Assistance screen with HOLCA.  

- **TC_Cluster_Alert_937_002**:  
  - **Action**: Use CAN signal `ASDRP_LnCntrWrnExtdIndReqAuth` to trigger "Lane Ending" alert.  
  - **Verify**: Alert displays in Monitor Tile.  

---

### **4. Scenario-Based Alerts**
- **Sensor/Lane Detection Issues**:  
  - **TC_Cluster_Alert_2148_Basic_RUN2**:  
    - **Condition**: Sensors cannot detect lane lines.  
    - **Verify**: Alert "Lane Centering Unavailable..." displays and deactivates after 3 seconds.  

- **Toll Booth/Hazard Alerts**:  
  - **TC_Cluster_Alert_724**:  
    - **Action**: Set CAN signal `Lane Centering Warning = Toll Booth Ahead`.  
    - **Verify**: "Autonomous Driving Unavailable - Toll Booth Ahead" displays.  

- **System Overheat/Wear Alerts**:  
  - **TC_MFC_Autonomous_Driving_Dev_0040**:  
    - **Action**: Trigger brake overheating and lane change cancellation.  
    - **Verify**: Alerts cycle every 12 seconds (e.g., "Brakes Overheated" and "Auto Lane Change Cancelled").  

---

### **5. Settings and Configuration Checks**
- **TC_Autonomous_Driving_Settings_101/100**:  
  - **Action**: Toggle "Auto Lane Change for Time" in settings with confirmation disabled.  
  - **Verify**: Snackbar "Settings Unavailable" appears for 11 seconds; toggle reverts to previous state.  

- **TC_Autonomous_Driving_Settings_123/122**:  
  - **Action**: Validate preference lists for features like Carpool Lanes and Open Lane Deceleration.  
  - **Verify**: Settings order and greying out align with configuration (e.g., Carpool Lanes 6th in list).  

- **TC_MFC_Autonomous_Driving_Settings_018**:  
  - **Action**: Adjust Open Lane Acceleration settings via MFC.  
  - **Verify**: Navigation and selection follow UI/UX guidelines.  

---

### **Key Parameters**
- **Power Modes Tested**: RUN, OFF, PROPULSION, ACCESSORY.  
- **Alert Cycling**: Alerts like "Brakes Overheated" and "Lane Change" cycle every 12 seconds under fault conditions.  
- **CAN Signals**: Critical signals include `Lane Centering Warning Extended Indication`, `ADAS Trim Level`, and `Brake Pad Worn Indication`.  
- **MFC Actions**: Knob nudges, button presses, and rotations validated across scenarios (e.g., Home, Previous, Next buttons).  

All tests emphasize precise signal handling, UI consistency, and alert timing under varying system states.