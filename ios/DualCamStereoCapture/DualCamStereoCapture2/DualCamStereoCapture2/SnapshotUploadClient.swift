import Foundation

struct SnapshotProcessingPreview: Decodable {
    let disposition: String
    let trustedDepthEligible: Bool
    let depthOverlayPath: String

    private enum CodingKeys: String, CodingKey {
        case disposition = "status"
        case trustedDepthEligible = "trusted_depth_eligible"
        case depthOverlayPath = "depth_overlay_url"
    }
}

final class SnapshotUploadClient {
    func upload(
        _ snapshot: StereoPhotoSnapshot,
        to endpoint: URL,
        completion: @escaping (Result<SnapshotProcessingPreview, Error>) -> Void
    ) {
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue(
            "multipart/form-data; boundary=\(boundary)",
            forHTTPHeaderField: "Content-Type"
        )
        request.httpBody = multipartBody(for: snapshot, boundary: boundary)

        URLSession.shared.dataTask(with: request) { data, response, error in
            if let error {
                completion(.failure(error))
                return
            }
            guard let httpResponse = response as? HTTPURLResponse,
                  (200..<300).contains(httpResponse.statusCode),
                  let data else {
                completion(.failure(URLError(.badServerResponse)))
                return
            }
            do {
                let preview = try JSONDecoder().decode(
                    SnapshotProcessingPreview.self, from: data
                )
                guard preview.disposition == "trusted"
                        || preview.disposition == "diagnostic" else {
                    completion(.failure(URLError(.cannotParseResponse)))
                    return
                }
                completion(.success(preview))
            } catch {
                completion(.failure(error))
            }
        }.resume()
    }

    private func multipartBody(
        for snapshot: StereoPhotoSnapshot,
        boundary: String
    ) -> Data {
        var body = Data()
        appendFile(
            snapshot.mainJPEG,
            field: "main_image",
            filename: "main.jpg",
            boundary: boundary,
            to: &body
        )
        appendFile(
            snapshot.ultrawideJPEG,
            field: "ultrawide_image",
            filename: "ultrawide.jpg",
            boundary: boundary,
            to: &body
        )
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append(
            "Content-Disposition: form-data; name=\"metadata\"\r\n\r\n"
                .data(using: .utf8)!
        )
        body.append(snapshot.metadataJSON)
        body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)
        return body
    }

    private func appendFile(
        _ data: Data,
        field: String,
        filename: String,
        boundary: String,
        to body: inout Data
    ) {
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append(
            "Content-Disposition: form-data; name=\"\(field)\"; filename=\"\(filename)\"\r\n"
                .data(using: .utf8)!
        )
        body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
        body.append(data)
        body.append("\r\n".data(using: .utf8)!)
    }
}
