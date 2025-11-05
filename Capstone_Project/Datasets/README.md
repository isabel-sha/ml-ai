# 📂 Datasets Folder

This folder contains the three datasets explored for this project. The data was originally sourced from **ServiceNow**, an IT Service Management (ITSM) platform. To protect sensitive information and maintain privacy, all identifiers and confidential fields have been anonymized or randomized, and certain date/time fields have been shifted. These modifications ensure that no real-world data can be traced, while still allowing the datasets to be used to test and demonstrate the concepts analyzed in this project. Any resemblance to actual events or individuals is entirely coincidental.

---

### Datasets Included in This Folder:

1. **Incident Data** (`modified_incident_metric.csv`)
   - **Number of Records:** 23,810
   - **Number of Columns:** 21

      <details>
         <summary>Click to expand/collapse column descriptions</summary>

      | Column Name           | Description |
      |-----------------------|-------------|
      | **Number**             | Unique identifier for each incident (anonymized as `INC######`) |
      | **Requested For**      | Name of the person who requested the incident (anonymized as a 6-character random string) |
      | **Business Unit**      | Department associated with the requester (anonymized as `Department` plus 3 random letters) |
      | **Assigned To**        | Person responsible for handling the incident (anonymized as a 6-character random string) |
      | **Assignment Group**   | Team responsible for the incident (anonymized as `Team` plus 3 random letters) |
      | **Active**             | Indicates whether the incident is active (`True` or `False`) |
      | **State**              | Current state of the incident (e.g., New, In Progress, On Hold, Resolved, Closed) |
      | **Priority**           | Incident priority level (e.g., Low, Medium, High) |
      | **Category**           | Incident category (e.g., Phishing, Software, Alert) |
      | **Channel**            | Communication channel (e.g., Chat, Email, Phone) |
      | **Opened**             | Timestamp when the incident was first opened (shifted) |
      | **Created**            | Timestamp when the incident was submitted (shifted) |
      | **Updated**            | Timestamp of the most recent update to the incident (shifted) |
      | **Resolved**           | Timestamp when the incident was resolved (shifted) |
      | **Closed**             | Timestamp when the incident was closed (shifted) |
      | **Reopen Count**       | Number of times the incident was reopened from a Resolved or Closed state |
      | **Reassignment Count** | Number of times the `Assigned To` person or `Assignment Group` changed |
      | **Change**             | Field that was changed (either `Assigned To` or `Assignment Group`) |
      | **Value**              | New value of the changed field (anonymized if applicable) |
      | **Start**              | Timestamp when the change occurred (shifted) |
      | **End**                | Timestamp when the change ended or was superseded (shifted) |
      </details>
2. **Request Data** (`modified_sc_task.csv`)
   - **Number of Records:** 31,963
   - **Number of Columns:** 13

      <details>
         <summary>Click to expand/collapse column descriptions</summary>
         
      | Column Name           | Description |
      |-----------------------|-------------|
      | **Number**             | Unique identifier for each request (anonymized as `TASK######`) |
      | **Requested For**      | Name of the person who requested (anonymized as a 6-character random string) |
      | **Business Unit**      | Department associated with the requester (anonymized as `Department` plus 3 random letters) |
      | **Assigned To**        | Person responsible for handling the request (anonymized as a 6-character random string) |
      | **Assignment Group**   | Team responsible for the request (anonymized as `Team` plus 3 random letters) |
      | **State**              | Current state of the request (e.g., New, In Progress, On Hold, Resolved, Closed) |
      | **Priority**           | Request priority level (e.g., Low, Medium, High) |
      | **Item**               | Item being requested (e.g., Monitor, Consultation, Software) |
      | **Opened**             | Timestamp when the request was first opened (shifted) |
      | **Created**            | Timestamp when the request was submitted (shifted) |
      | **Updated**            | Timestamp of the most recent update to the request (shifted) |
      | **Closed**             | Timestamp when the request was closed (shifted) |
      | **Reassignment Count** | Number of times the `Assigned To` person or `Assignment Group` changed |
      </details>
         
---
### Data Transformation Summary

The transformations described below were implemented in the script `data_transformer.ipynb`, which performs all anonymization, randomization, and timestamp shifting operations.

1. **Anonymization of Identifiers**:
   - Unique identifiers such as **Number** have been anonymized using randomized strings. For example, an incident number like `INC123456` may be replaced with a random identifier like `INC654321`.
   
2. **Randomization of Sensitive Information**:
   - Fields representing individuals such as **Assigned To** and **Requested For** have been replaced with randomized 6-character strings.
   - Fields representing organizational units such as **Business Unit** and **Assignment Group** have been generalized using a prefix plus a random string. For example, a business unit may be changed to `Department ABC`.

3. **Shifting of Date/Time Values**:
   - All timestamp fields such as **Created**, **Updated**, and **Resolved** have been randomly shifted in days, hours, minutes, and seconds. This preserves realistic timing patterns while removing any direct correlation to real-world events.
