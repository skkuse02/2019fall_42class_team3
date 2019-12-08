'use strict';

const database = require("./database");
const resource = require("./resource");
const credential = require("./credential");
const nodemailer = require("nodemailer");
const qrcode = require("qrcode");

const transporter = nodemailer.createTransport({
  service: "gmail",
  auth: {
    user: credential.mailUserId,
    pass: credential.mailUserPassword
  }
});

const generateQrcode = (url) => {
  const path = "/tmp/qrcode.png";
  qrcode.toFile(path, url);
  return path;
};

const generateToken = (seed="") => {
  const tokenLength = 15;
  const tokenArray = ['0','1','2','3','4','5','6','7','8','9',
                      'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
                      'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'];

  var token = "";
  for (var i = 1; i < tokenLength; i++) {
    token = token + tokenArray[Math.ceil(Math.random() * tokenArray.length) - 1];
  }

  return token;
};

const generateMailAuthenticationUrl = (userId, token) => {
  return resource.mailAuthenticateEndpoint + "?userId=" + userId + "&token=" + token;
}

const authenticateUser = async (userId) => {
  return await database.checkUserAuth(userId);
};

const authenticateMail = async (userId, token) => {
  const result = await database.authPendingAuthentication(userId, token);
  if (result.success == true) {
    return true;
  } else {
    return false;
  }
}

const sendAuthenticateMail = async (userId, userMail) => {
  const token = generateToken();
  const result = await database.registPendingAuthentication(userId, token);
  if (result.success == true) {
    const mailOptions = {
      from: credential.mailUserId,
      to: userMail,
      subject: "학우하구 인증 메일",
      text: "다음 링크를 클릭하셔서 메일을 인증해 주세요.\n" + generateMailAuthenticationUrl(userId, token)
    };

    try {
      const result = await new Promise((resolve, reject) => {
        transporter.sendMail(mailOptions, (err, info) => {
          resolve(info);
        });
      });
      console.log(result);
      if (Object.keys(result.accepted).length == 1) {
        return true;
      } else {
        return false;
      }
    } catch {
      return false;
    }
  } else {
    return false;
  }
};

module.exports = {
  generateQrcode,
  authenticateUser,
  authenticateMail,
  sendAuthenticateMail
}
