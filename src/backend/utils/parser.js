'use strict';

const getBody = (event) => {
  return ("body" in event ? JSON.parse(event.body) : event);
}

const getUserId = (event) => {
  return getBody(event).userRequest.user.id;
};

const getMessage = (event) => {
  return getBody(event).userRequest.utterance;
}

const getParameters = (event) => {
  return getBody(event).action.detailParams;
}

const getBlockId = (event) => {
  return getBody(event).userRequest.block.id;
}

const getKeys = (event) => {
  return event.queryStringParameters;
}

const getPrice = (price_string) => {
  return JSON.parse(price_string).amount
};

const getImageUrls = (image_list) => {
  return image_list.substring(5, image_list.length - 1).split(", ");
};

module.exports = {
  getBody,
  getUserId,
  getMessage,
  getParameters,
  getBlockId,
  getKeys,
  getPrice,
  getImageUrls
}
