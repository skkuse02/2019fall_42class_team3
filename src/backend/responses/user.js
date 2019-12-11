'use strict';

const builder = require("../utils/builder");
const resource = require("../utils/resource");

module.exports.userRegistration = () => {
  // Construct registration guide
  const guideThumbnail = builder.getThumbnail(resource.userRegistrationThumbnailUrl);
  const guideText = "대학생 중고 거래 서비스, 학우하구입니다!\n회원 가입을 원하시면 회원 가입 버튼을 눌러주세요.\n이미 회원 가입 신청을 하신 경우, 메일 인증을 완료하신 후 메일 인증 완료 버튼을 눌러주세요.";
  const guideRegistrationButton = builder.getButton("회원 가입", "block", "회원 가입", resource.registrationBlockId);
  const guideMainMenuButton = builder.getButton("메일 인증 완료", "block", "처음으로", resource.welcomeBlockId);
  const guideCard = builder.getBasicCard("", guideText, guideThumbnail, [guideRegistrationButton, guideMainMenuButton]);

  // Build response
  return builder.buildResponse([guideCard], false);
};

module.exports.userRegistrationSuccess = (userName) => {
  // Construct registration guide
  const guideThumbnail = builder.getThumbnail(resource.userRegistrationSuccessThumbnailUrl);
  const guideTitle = "회원 가입 성공";
  const guideDescription = "안녕하세요, " + userName + "님!\n학교 인증이 완료된 후 서비스를 이용하실 수 있습니다.\n메일 인증을 완료하신 후에 완료 버튼을 눌러주세요.";
  const guideMainMenuButton = builder.getButton("메일 인증 완료", "block", "처음으로", resource.welcomeBlockId);
  const guideCard = builder.getBasicCard(guideTitle, guideDescription, guideThumbnail, [guideMainMenuButton]);

  // Build response
  return builder.buildResponse([guideCard], false);
};


module.exports.userRegistrationFail = (errorMessage) => {
  // Construct registration guide
  const guideThumbnail = builder.getThumbnail(resource.userRegistrationFailThumbnailUrl);
  const guideTitle = "회원 가입 실패";
  const guideDescription = errorMessage;
  const guideCard = builder.getBasicCard(guideTitle, guideDescription, guideThumbnail);

  // Build response
  return builder.buildResponse([guideCard]);
};

module.exports.userInformation = (nickname, school_name, school_mail, openprofile, reliability_score) => {
  // Construct user information
  const infoTitle = "회원 정보";
  const infoNicknameList = builder.getListItem(nickname, "닉네임");
  const infoSchoolNameList = builder.getListItem(school_name, "학교 이름");
  const infoSchoolMailList = builder.getListItem(school_mail, "학교 이메일");
  const infoOpenProfileList = builder.getListItem(openprofile, "오픈프로필 주소");
  const infoReliabilityScoreList = builder.getListItem(reliability_score, "신뢰도");
  const infoOpenProfileButton = builder.getButton("오픈프로필", "webLink", openprofile);
  const infoWithdrawButton = builder.getButton("회원 탈퇴", "block", "회원 탈퇴", resource.userWithdrawWarningBlockId);
  const infoCard = builder.getListCard(infoTitle, "", [infoNicknameList, infoSchoolNameList, infoSchoolMailList, infoOpenProfileList, infoReliabilityScoreList],
                                                      [infoOpenProfileButton, infoWithdrawButton]);

  // Build response
  return builder.buildResponse([infoCard]);
};

module.exports.userAuthenticationSuccess = () => {
  return builder.buildAWSResponse("Mail authentication success!");
};

module.exports.userAuthenticationFail = () => {
  return builder.buildAWSResponse("Mail authentication fail!");
};

module.exports.userOpenprofileSuccess = (openprofile) => {
  const listItem = builder.getListItem(openprofile, "오픈프로필 링크");
  const listCard = builder.getListCard("오픈프로필 등록 성공", "", [listItem]);
  return builder.buildResponse([listCard]);
};

module.exports.userWithdrawWarning = () => {
  const okButton = builder.getButton("확인", "block", "확인", resource.userWithdrawBlockId, {action: "ok"});
  const cancelButton = builder.getButton("취소", "block", "취소", resource.userWithdrawBlockId, {action: "cancel"});
  const basicCard = builder.getBasicCard("정말 회원을 탈퇴하시겠습니까?", "", "", [okButton, cancelButton]);
  return builder.buildResponse([basicCard], false);
};

module.exports.userWithdrawOk = () => {
  const resultThumbnail = builder.getThumbnail(resource.itemRegistrationSuccessThumbnailUrl);
  const basicCard = builder.getBasicCard("회원에서 탈퇴했습니다.", "", resultThumbnail);
  return builder.buildResponse([basicCard]);
};

module.exports.userWithdrawCancel = () => {
  const textCard = builder.getSimpleText("회원 탈퇴를 취소했습니다.");
  return builder.buildResponse([textCard]);
};

module.exports.userReport = (user_name) => {
  const textCard = builder.getSimpleText(user_name + "(을)를 신고했습니다.");
  return builder.buildResponse([textCard]);
};

module.exports.userContract = (contracts, userId) => {
  var text = "총 거래 수: " + contracts.length + "건";
  for (var key in contracts) {
    const contract = contracts[key];
    if (contract.sell_id == userId) {
      text += "\n판매 - " + contract.item_name;
    } else if (contract.buy_id == userId) {
      text += "\n구매 - " + contract.item_name;
    }
  }
  return builder.buildResponse([builder.getSimpleText(text)]);
};
