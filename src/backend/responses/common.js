'use strict';

const builder = require("../utils/builder");
const resource = require("../utils/resource");

module.exports.welcome = () => {
  // Construct buyer card
  const buyerThumbnail = builder.getThumbnail(resource.buyerThumbnailUrl);
  const buyerSearchTextButton = builder.getButton("상품 카테고리 검색", "block", "상품 카테고리 검색", resource.buyerSearchCategoryBlockId);
  const buyerSearchImageButton = builder.getButton("상품 키워드 검색", "block", "상품 키워드 검색", resource.buyerSearchNameBlockId);
  const buyerContractButton = builder.getButton("구매자 거래 체결", "block", "구매자 거래 체결", resource.buyerContractBlockId);
  const buyerCardBody = builder.getBasicCardBody("구매", "상품을 구매합니다.", buyerThumbnail, [buyerSearchTextButton, buyerSearchImageButton, buyerContractButton]);

  // Construct seller card
  const sellerThumbnail = builder.getThumbnail(resource.sellerThumbnailUrl);
  const sellerItemRegistButton = builder.getButton("상품 등록", "block", "상품 등록", resource.sellerItemRegistBlockId);
  const sellerItemListButton = builder.getButton("상품 목록", "block", "상품 목록", resource.sellerItemListBlockId);
  const sellerContractButton = builder.getButton("판매자 거래 체결", "block", "판매자 거래 체결", resource.sellerContractBlockId);
  const sellerCardBody = builder.getBasicCardBody("판매", "상품을 판매합니다.", sellerThumbnail, [sellerItemRegistButton, sellerItemListButton, sellerContractButton]);

  // Construct user information card
  const userInfoThumbnail = builder.getThumbnail(resource.userInfoThumbnailUrl);
  const userInfoQueryButton = builder.getButton("회원 정보 조회", "block", "회원 정보 조회", resource.userInfoQueryBlockId);
  const userInfoOpenProfileButton = builder.getButton("오픈프로필 링크 수정", "block", "오픈프로필 링크 수정", resource.userInfoOpenProfileBlockId);
  const userInfoTimeTableButton = builder.getButton("시간표 수정", "block", "시간표 수정", resource.userInfoTimeTableBlockId);
  const userInfoCardBody = builder.getBasicCardBody("회원 정보", "회원 정보를 조회하거나 수정합니다.", userInfoThumbnail, [userInfoQueryButton, userInfoOpenProfileButton, userInfoTimeTableButton]);

  // Build response
  return builder.buildResponse([builder.getCarousel("basicCard", [buyerCardBody, sellerCardBody, userInfoCardBody])]);
};
