const nodemailer = require("nodemailer");

function executeQuery(query) {
    return new Promise((resolve, reject) => {
      connection.query(query, (err, results) => {
        if (err) {
          console.log(err);
          reject(err);
        } else {
          // console.log(results)
          resolve(results);
        }
      });
    });
  }
  
  
  
  const transporter = nodemailer.createTransport({
    host: "smtp.gmail.com",
    port: "587",
    service: false,
    auth: {
      user: "er.priyankabarate@gmail.com",
      pass: "alko kysr cqnx vymz",
    },
  });

  const sendOtp = (otp, email, name) => {

    let mailOptions = {
      from: "",
      to: email,
      subject: "OTP for Password Setup – Marworx Chatbot",
      html: `<!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>CRM</title>
            </head>
            <body style=" font-family:Arial, Helvetica, sans-serif;font-size: 14px;">
                <div style="min-width: 320px; max-width: 550px; margin: 0px auto;">
                    <P style="color: #252729; margin: 0; ">Dear ${name},</p>
                    <div style="background-color: white; padding: 25px 20px;color:#4f5459;">
                        <p style="color:#4f5459; margin: 0; ">Use the OTP below to set your password for your <strong>Chatbot  </strong> account:
                        </p>
                        <p style="color: #252729; margin: 0;">🔹<strong> OTP</strong>:${otp} </p> 
                        <p style="color:#4f5459; margin: 0;">This OTP is valid for a 5 minutes. If you did not request this, please ignore this email.</p>
                    </div>
                </div>
            </body>
            
            </html>`,
    };
    transporter.sendMail(mailOptions, (error, info) => {
      if (error) {
        console.error("Error sending email:", error);
        return { error: "Error", error };
      } else {
        console.log("Email sent:", info.response);
        return { success: "Email sent:", response: info.response };
      }
    });
  };

  module.exports = {
    sendOtp
  }