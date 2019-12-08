'use strict';

const builder = require("../utils/builder");
const resource = require("../utils/resource");

module.exports.itemRegistrationSuccess = (itemName, itemDetail, itemPrice, imageUrl) => {
  // Construct registration guide
  const resultThumbnail = builder.getThumbnail(imageUrl);
  const resultTitle = itemName;
  const resultDescription = itemDetail;
  const resultPrice = itemPrice;
  const resultMainMenuButton = builder.getButton("처음으로", "block", "처음으로", resource.welcomeBlockId);
  const resultCard = builder.getCommerceCard(resultTitle, resultDescription, resultPrice, resultThumbnail, null, [resultMainMenuButton]);

  // Build response
  return builder.buildResponse([resultCard]);
};

module.exports.itemRegistrationFail = (errorMessage) => {
  // Construct registration guide
  const resultThumbnail = builder.getThumbnail(resource.itemRegistrationFailThumbnailUrl);
  const resultTitle = "상품 등록 실패";
  const resultDescription = errorMessage;
  const resultMainMenuButton = builder.getButton("처음으로", "block", "처음으로", resource.welcomeBlockId);
  const resultCard = builder.getBasicCard(resultTitle, resultDescription, resultThumbnail, [resultMainMenuButton]);

  // Build response
  return builder.buildResponse([resultCard]);
};

module.exports.itemListSuccess = (itemList) => {
  // Construct commerce cards
  const bodys = [];
  for (var key in itemList) {
    if (key > 9) {
      break;
    }
    const item = itemList[key];
    const resultThumbnail = builder.getThumbnail(item.item_image[0]);
    const resultTitle = item.item_name;
    const resultDescription = item.item_detail;
    const resultPrice = item.item_price;
    const resultNickname = (item.nickname != undefined ? item.nickname : null);
    const resultOpenprofile = item.openprofile;
    const resultMainMenuButton = builder.getButton("처음으로", "block", "처음으로", resource.welcomeBlockId);
    bodys.push(builder.getCommerceCardBody(resultTitle, resultDescription, resultPrice, resultThumbnail, resultNickname, [resultMainMenuButton]));
  }

  // Build response
  return builder.buildResponse([builder.getCarousel("commerceCard", bodys)]);
};

module.exports.itemListFail = (errorMessage) => {
  // Construct result card
  const resultThumbnail = builder.getThumbnail(resource.itemRegistrationFailThumbnailUrl);
  const resultTitle = "상품 조회 실패";
  const resultDescription = errorMessage;
  const resultMainMenuButton = builder.getButton("처음으로", "block", "처음으로", resource.welcomeBlockId);
  const resultCard = builder.getBasicCard(resultTitle, resultDescription, resultThumbnail, [resultMainMenuButton]);

  // Build response
  return builder.buildResponse([resultCard]);
};
