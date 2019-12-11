'use strict';

const parser = require("../utils/parser");
const database = require("../utils/database");
const authenticator = require("../utils/authenticator");
const responseTemplate = require("../responses/template");

const registration = async (event) => {
  const userId = parser.getUserId(event);
  const parameters = parser.getParameters(event);
  const item_category = parameters["item_category"].value;
  const item_name = parameters["item_name"].origin;
  const item_price = parser.getPrice(parameters["item_price"].value);
  const item_detail = parameters["item_detail"].origin;
  const item_image = parser.getImageUrls(parameters["item_image"].origin);

  const result = await database.registNewItem(userId, item_category, item_name, item_price, item_detail, item_image);
  if (result.success == true) {
    return responseTemplate.itemRegistrationSuccess(item_name, item_detail, item_price, item_image[0]);
  } else {
    return responseTemplate.itemRegistrationFail("");
  }
}

const list = async (event) => {
  const userId = parser.getUserId(event);
  const mode = parser.getExtras(event)["mode"];

  const result = await database.getUserItem(userId);
  if (result == null) {
    return responseTemplate.itemListFail("상품 조회에 실패했습니다.");
  } else if (result.length == 0) {
    return responseTemplate.itemListFail("등록된 상품이 없습니다.");
  } else {
    return responseTemplate.itemListSuccess(result, mode);
  }
};

const searchCategory = async (event) => {
  const parameters = parser.getParameters(event);
  const item_category = parameters["item_category"].value;

  const result = await database.getCategory(item_category);
  if (result == null) {
    return responseTemplate.itemListFail("상품 조회에 실패했습니다.");
  } else if (result.length == 0) {
    return responseTemplate.itemListFail("등록된 상품이 없습니다.");
  } else {
    return responseTemplate.itemListSuccess(result, "search");
  }
};

const searchKeyword = async (event) => {
  const parameters = parser.getParameters(event);
  const item_category = parameters["item_keyword"].origin;

  const result = await database.getKeyword(item_category);
  if (result == null) {
    return responseTemplate.itemListFail("상품 조회에 실패했습니다.");
  } else if (result.length == 0) {
    return responseTemplate.itemListFail("등록된 상품이 없습니다.");
  } else {
    return responseTemplate.itemListSuccess(result, "search");
  }
};

const detail = async (event) => {
  const extras = parser.getExtras(event);
  const mode = extras["mode"];
  const userId = extras["userId"];
  const itemId = extras["itemId"];

  const item = await database.getItem(itemId);
  if (item != null) {
    const user = (mode == "search" ? await database.getUser(userId) : null);
    if (user != null || mode == "list") {
      return responseTemplate.itemDetail(item, mode, user);
    } else {
      return responseTemplate.itemListFail("상품 조회에 실패했습니다.");
    }
  } else {
    return responseTemplate.itemListFail("상품 조회에 실패했습니다.");
  }
};

const sellerContract = async (event) => {
  const userId = parser.getUserId(event);
  const extras = parser.getExtras(event);
  const itemId = extras["itemId"];

  const qrcode = authenticator.getContractQrcode(userId, itemId);
  const token = qrcode.token;
  const qrcodePaths = await database.uploadImages("contract/" + token + "/", [qrcode.qrcodePath]);

  if (qrcodePaths == null) {
    return responseTemplate.processFail("상품 판매 실패", "상품 판매 등록에 실패했습니다.");
  } else {
    const result = await database.registTempDeal(userId, itemId, token);
    if (result.success == true) {
      return responseTemplate.itemSellerContractSuccess(qrcodePaths[0], itemId);
    } else {
      return responseTemplate.processFail("상품 판매 실패", "상품 판매 등록에 실패했습니다.");
    }
  }
};

const buyerContract = async (event) => {
  const userId = parser.getUserId(event);
  const parameters = parser.getParameters(event);
  const item_data = parser.getQrcodeData(parameters["item_qrcode"].value);
  const itemUserId = item_data.userId;
  const itemId = item_data.itemId;

  const item = await database.getItem(itemId);
  if (item != null) {
    const result = await database.registNewDeal(itemUserId, userId, itemId, item.item_name);
    if (result.success == true) {
      return responseTemplate.itemBuyerContractSuccess(item);
    } else {
      return responseTemplate.processFail("상품 구매 실패", "상품 구매 등록에 실패했습니다.");
    }
  } else {
    return responseTemplate.processFail("상품 구매 실패", "상품 구매 등록에 실패했습니다.");
  }
};

const deleteWarning = async (event) => {
  const extras = parser.getExtras(event);
  const itemId = extras["itemId"];
  const item_image = extras["item_image"];

  return responseTemplate.itemDeleteWarning(itemId, item_image);
};

const deleteOk = async (event) => {
  const extras = parser.getExtras(event);
  const itemId = extras["itemId"];
  const action = extras["action"];

  if (action == "ok") {
    const result = await database.deleteItem(itemId);
    if (result.success == true) {
      return responseTemplate.itemDeleteOk();
    } else {
      return responseTemplate.processFail("상품 삭제 실패", "상품 삭제에 실패했습니다.");
    }
  } else {
    return responseTemplate.itemDeleteCancel();
  }
};

module.exports = {
  registration,
  list,
  searchCategory,
  searchKeyword,
  detail,
  sellerContract,
  buyerContract,
  deleteWarning,
  deleteOk
};
