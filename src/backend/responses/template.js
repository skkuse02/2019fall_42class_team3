'use strict';

const commonResponse = require("./common");
const userResponse = require("./user");
const itemResponse = require("./item");

module.exports = {
  welcome: commonResponse.welcome,
  processFail: commonResponse.processFail,
  userRegistration: userResponse.userRegistration,
  userRegistrationSuccess: userResponse.userRegistrationSuccess,
  userRegistrationFail: userResponse.userRegistrationFail,
  userInformation: userResponse.userInformation,
  userAuthenticationSuccess: userResponse.userAuthenticationSuccess,
  userAuthenticationFail: userResponse.userAuthenticationFail,
  itemRegistrationSuccess: itemResponse.itemRegistrationSuccess,
  itemRegistrationFail: itemResponse.itemRegistrationFail,
  itemListSuccess: itemResponse.itemListSuccess,
  itemListFail: itemResponse.itemListFail,
  itemDetail: itemResponse.itemDetail,
  itemSellerContractSuccess: itemResponse.itemSellerContractSuccess,
  itemBuyerContractSuccess: itemResponse.itemBuyerContractSuccess,
  itemDeleteWarning: itemResponse.itemDeleteWarning,
  itemDeleteOk: itemResponse.itemDeleteOk,
  itemDeleteCancel: itemResponse.itemDeleteCancel,
  userOpenprofileSuccess: userResponse.userOpenprofileSuccess,
  userWithdrawWarning: userResponse.userWithdrawWarning,
  userWithdrawOk: userResponse.userWithdrawOk,
  userWithdrawCancel: userResponse.userWithdrawCancel,
  userReport: userResponse.userReport,
  userContract: userResponse.userContract
};
