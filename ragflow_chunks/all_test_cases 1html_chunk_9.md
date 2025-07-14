The document outlines test cases for validating vehicle lane assistance (LKA, BZSA, HoLCA, LDW) and autonomous driving systems, emphasizing functional safety, UI consistency, and signal interactions.  

### **Lane Assistance System Testing**  
**Key Objectives**:  
- Verify lane change alerts (e.g., "Autonomous Driving Lane Change Cancelled Left") during power mode transitions (RUN, PROPULSION, ACC) and trailer-related limitations (e.g., **Lane Change Alert Limited by Trailer CAL** = Enabled/Disabled).  
- Validate LKA/BZSA UI toggles and signal outputs (e.g., **Lane Keep Assist Selected Virtual Control Request Signal** = On/Off) under system states like availability, unavailability, or failure (e.g., **Lane Detection Warning and Control Feature State** = FAILED).  
- Ensure alerts deactivate/reactivate during power mode shifts (e.g., OFF→ACC) and align with trim levels (e.g., **Advanced Driver Assist Systems Trim Level** = Autonomous Driving).  
- Test UI functionality in infotainment-off (NoIVI) scenarios, including snackbar/dialog displays and control availability.  
- Validate critical signals (e.g., **Vehicle Speed Average Driven Authenticated Signal** >8 km/h) and thresholds (e.g., **Jackknife Threshold Hysteresis CAL** = 5°).  

**Key Components**:  
- Signals: **Lane Keep Assist/BZSA Virtual Control Request Signals**, **System Power Mode Authenticated Signal**, and **Lane Detection States** (STANDBY, READY TO ASSIST, ALERT AND CONTROL).  
- Alerts: Visual (amber/green icons), haptic feedback, and priority-based alerts (e.g., high-priority "Autonomous Driving Lane Change Unable Right" overriding lower ones).  
- Edge Cases: System behavior during sleep/wakeup cycles, battery reconnection, and invalid signal states (e.g., **Vehicle Speed Authenticated Invalid Signal** = FALSE).  

**Notable Test Cases**:  
- **TC 652**: Confirms "Lane Change Alert Limited by Trailer" alert visibility when CAL = Enabled and power mode = RUN.  
- **TC 660**: Validates BZSA checkbox state in IVI screens under specific power and feature states.  
- **TC 835/TC 908**: Test snackbars in NoIVI mode (e.g., "Blind Zone Steering Assist unavailable" dismisses in 3 seconds).  
- **TC 1117**: Ensures alerts deactivate correctly in ACC power mode when Lane Centering Warning = No Indication.  

**Outcomes**:  
- System compliance with safety standards, dynamic alert responses to driving conditions (speed, power modes, trailer presence), and UI consistency across configurations.  

---

### **Autonomous Driving System Testing**  
**Core Focus**:  
- **Alert Verification**: Test alerts (e.g., "Autonomous Driving Unavailable – Trailer Over Weight Limit") triggered by CAN signals (e.g., **ADAS Trim Level** = "Autonomous Driving") and system states (e.g., power modes, sensor inputs).  
- **System Interactions**: Use tools like **Vehicle Spy** to simulate CAN signals (e.g., *"Lane Centering Warning Extended Indication Request"* = "Trailer Narrow Lane") and validate responses.  
- **UI/UX Validation**: Ensure alerts display correctly on Cluster Display/Center Stack/HUD, with dynamic lane guidance updates (e.g., 8+ lanes rendered across 11 slots) and visual cues (green/red indicators).  
- **Edge Cases**: Test behavior under "No Road Information," poor weather, GPS loss, and power mode shifts (e.g., deactivating *"Lane Centering Assistance"* in OFF mode).  

**Technical Components**:  
- **ADAS ECU Communication**: Verify IVI-ADAS ECU interactions via CAN networks, including authenticated signals (e.g., *"Serial Data 34 Protected"*).  
- **Signal Authentication**: Secure responses to signals like *"Lane Centering Control Indication Request."*  

**Tools**:  
- **Vehicle Spy**: Simulate CAN signals for testing (e.g., *"Adaptive Cruise Control Disengaging Indication"*).  
- **Mock FSA**: Replicate user preferences (e.g., enabling/disabling "Lane Change Frequency").  

**Outcomes**:  
- Alerts and indicators align with signal states (e.g., *"Lane Departure Warning Unavailable"* when sensors are offline).  
- UI elements (e.g., lane guidance animations) adapt dynamically to driving conditions and user settings.  
- System reliability under edge cases (e.g., GPS loss, sensor unavailability) and adherence to design specs (e.g., lane change frequency persistence).  

This structured testing ensures robust validation of safety, precision in signal handling, and UI consistency across scenarios.