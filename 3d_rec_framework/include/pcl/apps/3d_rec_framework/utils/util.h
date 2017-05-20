#pragma once
#include <string>
#include <algorithm>
#include <Windows.h>

// process the path to get the right format 
std::string getDirectoryPath(std::string path) {
	std::replace(path.begin(), path.end(), '\\', '/');
	int lastSlashIndex = path.find_last_of('/', (int)path.size());
	if (lastSlashIndex < (int)path.size() - 1)
		path += "/";
	return path;
}



int countNumberOfFilesInDirectory(std::string inputDirectory, const char* fileExtension) {
	char search_path[300];
	WIN32_FIND_DATA fd;
	sprintf_s(search_path, fileExtension, inputDirectory.c_str());
	HANDLE hFind = ::FindFirstFile(search_path, &fd);

	//count the number of OCT frames in the folder
	int count = 0;
	if (hFind != INVALID_HANDLE_VALUE)
	{
		do
		{
			if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
			{

				count++;

			}
		} while (::FindNextFile(hFind, &fd));
		::FindClose(hFind);
	}
	return count;
}
