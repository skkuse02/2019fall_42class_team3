'use strict';

const parser = require("../utils/parser");
const database = require("../utils/database");
const authenticator = require("../utils/authenticator");
const responseTemplate = require("../responses/template");

var mail;

const registration = async (event) => {
  const userId = parser.getUserId(event);
  const parameters = parser.getParameters(event);
  const nickname = parameters["nickname"].origin;
  const school_name = parameters["school_name"].value;
  const school_mail = parameters["school_mail"].origin;

  if (authenticator.authenticateUser(userId) == true) {
    return responseTemplate.userRegistrationFail("이미 등록된 사용자입니다.");
  } else {
    const mail_success = await authenticator.sendAuthenticateMail(userId, school_mail);
    if (mail_success == true) {
      const result = await database.registNewUser(userId, nickname, school_name, school_mail);
      if (result.success == true) {
        return responseTemplate.userRegistrationSuccess(nickname);
      } else {
        return responseTemplate.userRegistrationFail(result.message);
      }
    } else {
      return responseTemplate.userRegistrationFail("인증 메일 발송에 실패하였습니다.");
    }
  }
}

const authentication = async (event) => {
  const keys = parser.getKeys(event);
  const userId = keys["userId"];
  const token = keys["token"];

  const success = await authenticator.authenticateMail(userId, token);
  if (success == true) {
    return responseTemplate.userAuthenticationSuccess();
  } else {
    return responseTemplate.userAuthenticationFail();
  }
};

const information = async (event) => {
  const userId = parser.getUserId(event);

  const user = await database.getUser(userId);
  if (user == null) {
    return responseTemplate.userRegistration();
  } else {
    const openprofile = (user.openprofile == null ? "-" : user.openprofile);
    return responseTemplate.userInformation(user.nickname, user.school_name, user.school_mail, openprofile, user.reliability_score);
  }
};

const openprofile = async (event) => {
  const userId = parser.getUserId(event);
  const parameters = parser.getParameters(event);
  const user_openprofile = parameters["user_openprofile"].origin;

  const result = await database.setOpenprofile(userId, user_openprofile);
  if (result.success == true) {
    return responseTemplate.userOpenprofileSuccess(user_openprofile);
  } else {
    return responseTemplate.processFail("오픈프로필 등록 실패", "오픈프로필 등록에 실패하였습니다.");
  }
};

const withdrawWarning = async (event) => {
  return responseTemplate.userWithdrawWarning();
};

const withdraw = async (event) => {
  const userId = parser.getUserId(event);
  const extras = parser.getExtras(event);
  const action = extras["action"];

  if (action == "ok") {
    const result = await database.deleteUser(userId);
    if (result.success == true) {
      return responseTemplate.userWithdrawOk();
    } else {
      return responseTemplate.processFail("회원 탈퇴 실패", "회원 탈퇴에 실패하였습니다.");
    }
  } else {
    return responseTemplate.userWithdrawCancel();
  }
};

const report = async (event) => {
  const extras = parser.getExtras(event);
  const targetUserId = extras["targetUserId"];
  const targetNickname = extras["targetNickname"];

  const result = await database.reportUser(targetUserId);
  if (result.success == true) {
    return responseTemplate.userReport(targetNickname);
  } else {
    return responseTemplate.processFail("신고 실패", "신고에 실패하였습니다.");
  }
};

const contract = async (event) => {
  var userId = parser.getUserId(event);
  const extras = parser.getExtras(event);
  if ("targetUserId" in extras) {
    userId = extras["targetUserId"];
  }

  const contracts = await database.getTransaction(userId);
  if (contracts == null) {
    return responseTemplate.processFail("거래 기록 조회 실패", "거래 기록 조회에 실패하였습니다.");
  } else {
    return responseTemplate.userContract(contracts, userId);
  }
};

module.exports = {
  registration,
  authentication,
  information,
  openprofile,
  withdrawWarning,
  withdraw,
  report,
  contract
};
