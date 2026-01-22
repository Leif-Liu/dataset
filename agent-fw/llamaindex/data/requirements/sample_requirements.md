Sample Requirement Document

Project: User Account Management

1. Registration
- The system shall allow users to register with email and password.
- Password must be at least 8 characters and contain letters and numbers.
- After registration, send a verification email with a one-time link.

2. Login
- The system shall allow users to login with verified email and password.
- After 5 failed attempts, lock the account for 10 minutes.
- Provide "forgot password" to reset via email.

3. Profile
- Users can view and edit profile information (name, phone, avatar).
- Changes must be saved and reflected immediately after update.

4. Security
- All passwords must be stored securely (hashed).
- Sessions expire after 30 minutes of inactivity.

